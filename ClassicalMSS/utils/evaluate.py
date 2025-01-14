# Copyright (c) Victor Zhou
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Structure of the project is learned from https://github.com/facebookresearch/demucs

from concurrent import futures
import logging

from dora.log import LogProgress
import numpy as np
import musdb
import museval
import torch as th

import sys
from pathlib import Path
root = Path(__file__).parent.parent
sys.path.append(str(root))

from .apply import apply_model
from .audio import convert_audio, save_audio
from .utils import DummyPoolExecutor

logger = logging.getLogger(__name__)

def new_sdr(references, estimates):
    """
    Compute the SDR according to the MDX challenge definition.
    Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
    """
    assert references.dim() == 4
    assert estimates.dim() == 4
    delta = 1e-7  # avoid numerical errors
    num = th.sum(th.square(references), dim=(2, 3))
    den = th.sum(th.square(references - estimates), dim=(2, 3))
    num += delta
    den += delta
    scores = 10 * th.log10(num / den)
    return scores

def eval_track(references, estimates, win, hop, compute_sdr=True):
    # Shape of the references/estimates input should be (sources, channels, samples), transpose to (sources, samples, channels)
    references = references.transpose(1, 2).double()
    estimates = estimates.transpose(1, 2).double()

    # The new_sdr function expects a shape of (batch_size, sources, samples, channels), so we add a dimension "1" at place 0
    # The new_sdr function returns a shape of (batch_size, sources), since we only have one batch, we take the sources' dimension
    new_sdr_scores = new_sdr(references.cpu()[None], estimates.cpu()[None])[0]

    if not compute_sdr:
        return None, new_sdr_scores
    else:
        references = references.numpy()
        estimates = estimates.numpy()
        # museval.metrics.bss_eval() (Blind Source Separation Evaluation) returns a tuple of:
        # 1. SDR (Signal to Distortion Ratio)
        # 2. ISR (Image to Spatial distortion Ratio)
        # 3. SIR (Signal to Interference Ratio)
        # 4. SAR (Signal to Artifacts Ratio)
        # 5. A permutation array (Only needed when using Permutation Invariant Training, PIT, but here the order of the sources is fixed)
        # So we don't need the last element of the returned tuple
        scores = museval.metrics.bss_eval(
            references, estimates,
            compute_permutation=False,
            window=win,
            hop=hop,
            # "framewise_filters" compute different filter coefficients (which is used to determine the portion of the reference that is in the estimate)
            # for each frame (window). This will make the calculation less stable and slower, so is generally set to False in MSS evaluation
            framewise_filters=False,
            bsseval_sources_version=False # Uses the newer BSS Eval v4 instead of the older BSS Eval Sources version
        )[:-1]
        # new_sdr_scores is a shape of (sources), while scores is a shape of (4, sources, windows)
        # 4 is the number of metrics (SDR, ISR, SIR, SAR)
        return scores, new_sdr_scores


def evaluate(solver, compute_sdr=False):
    """
    Evaluate model using museval.
    compute_sdr=False means using only the MDX definition of the SDR, which
    is much faster to evaluate.
    """

    args = solver.args
    output_dir = solver.folder / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    json_folder = solver.folder / "results/test"
    json_folder.mkdir(exist_ok=True, parents=True)

    # we load tracks from the original musdb set
    if args.test.nonhq is None:
        test_set = musdb.DB(args.dset.musdb, subsets=["test"], is_wav=True)
    else:
        test_set = musdb.DB(args.test.nonhq, subsets=["test"], is_wav=False)
    src_rate = args.dset.musdb_samplerate

    eval_device = 'cpu'

    model = solver.model
    win = int(1. * model.samplerate) # Window size
    hop = int(1. * model.samplerate) # Hop size

    indexes = range(len(test_set)) # No distributed training yet
    indexes = LogProgress(logger, indexes, updates=args.misc.num_prints,
                          name='Eval') # Progress bar
    pendings = []

    # If parallelization is needed, use the ProcessPoolExecutor;
    # otherwise, use the DummyPoolExecutor (which does basically nothing)
    pool = futures.ProcessPoolExecutor if args.test.workers else DummyPoolExecutor
    with pool(args.test.workers) as pool:
        for index in indexes:
            track = test_set.tracks[index]

            mix = th.from_numpy(track.audio).t().float()
            if mix.dim() == 1:
                mix = mix[None]
            mix = mix.to(solver.device)
            ref = mix.mean(dim=0)  # mono mixture
            mix = (mix - ref.mean()) / ref.std() # Normalize the mix to have a better estimate
            mix = convert_audio(mix, src_rate, model.samplerate, model.audio_channels)
            estimates = apply_model(model, mix[None],
                                    shifts=args.test.shifts, split=args.test.split,
                                    overlap=args.test.overlap)[0]
            estimates = estimates * ref.std() + ref.mean() # Denormalize the estimates
            estimates = estimates.to(eval_device)

            # Shape of the references should be (sources, channels, samples)
            references = th.stack(
                [th.from_numpy(track.targets[name].audio).t() for name in model.sources])
            if references.dim() == 2: # If the ref is mono
                references = references[:, None] # Add a dimension at place 1
            references = references.to(eval_device)
            references = convert_audio(references, src_rate,
                                       model.samplerate, model.audio_channels)
            if args.test.save:
                folder = solver.folder / "wav" / track.name
                folder.mkdir(exist_ok=True, parents=True)
                for name, estimate in zip(model.sources, estimates):
                    save_audio(estimate.cpu(), folder / (name + ".mp3"), model.samplerate)

            # "start evaluating this track in the background and I'll check the results later"
            # pool.submit takes a function (eval_track) and its arguments, and schedules it to run in a separate process
            # It returns a Future object that is stored along with the track name
            # Future.result() will get the actual results when they are ready
            pendings.append((track.name, pool.submit( 
                eval_track, references, estimates, win=win, hop=hop, compute_sdr=compute_sdr)))

        pendings = LogProgress(logger, pendings, updates=args.misc.num_prints,
                               name='Eval (BSS)') # Blind Source Separation
        tracks = {}
        for track_name, pending in pendings:
            pending = pending.result()
            scores, nsdrs = pending # nsdrs stands for "new SDRs" which is the SDR computed by the new_sdr function (the MDX challenge definition)
            tracks[track_name] = {}
            for idx, target in enumerate(model.sources):
                tracks[track_name][target] = {'nsdr': [float(nsdrs[idx])]}
            if scores is not None:
                (sdr, isr, sir, sar) = scores
                for idx, target in enumerate(model.sources):
                    values = {
                        "SDR": sdr[idx].tolist(),
                        "SIR": sir[idx].tolist(),
                        "ISR": isr[idx].tolist(),
                        "SAR": sar[idx].tolist()
                    }
                    tracks[track_name][target].update(values)

        # No distributed training yet
        all_tracks = tracks

        result = {}
        metric_names = next(iter(all_tracks.values()))[model.sources[0]] # It is only used to get the metric names (Since every track/soureces has the same metrics)
        for metric_name in metric_names:
            avg = 0
            avg_of_medians = 0
            for source in model.sources:
                medians = [
                    np.nanmedian(all_tracks[track][source][metric_name]) # Median across all windows of the same track
                    for track in all_tracks.keys()]
                mean = np.mean(medians) # Mean across all tracks
                median = np.median(medians) # Median across all tracks
                result[metric_name.lower() + "_" + source] = mean
                result[metric_name.lower() + "_med_" + source] = median
                avg += mean / len(model.sources) # Mean across all sources
                avg_of_medians += median / len(model.sources) # Mean of median across all sources
            result[metric_name.lower()] = avg
            result[metric_name.lower() + "_med"] = avg_of_medians
        return result
