import numpy as np
import mne


def find_bad_epochs(
    epochs,
    threshold,
    tmin=None,
    tmax=None,
    measure="peak-to-peak",
    picks=None,
    verbose=None,
):
    if tmin is None:
        tmin = epochs.reject_tmin

    if tmax is None:
        tmax = epochs.reject_tmax

    data = epochs.get_data(picks=picks, tmin=tmin, tmax=tmax)

    if picks is not None:
        ch_names = epochs.copy().pick(picks=picks).ch_names
    else:
        ch_names = epochs.ch_names

    if measure == "absolute":
        mask = np.abs(data) > threshold

        bad_channels = mask.any(axis=2)
    elif measure == "zscore":
        mean = data.mean(axis=(0, 2), keepdims=True)
        std = data.std(axis=(0, 2), keepdims=True)

        zscore = (data - mean) / std
        mask = np.abs(zscore) > threshold

        bad_channels = mask.any(axis=2)
    elif measure == "peak-to-peak":

        bad_channels = np.ptp(data, axis=2) > threshold

    else:
        raise ValueError(f"Unknown measure: {measure}")

    indices = np.where(bad_channels.any(axis=1))[0]

    reasons = {
        int(i): [ch_names[j] for j in np.where(bad_channels[i])[0]] for i in indices
    }

    return indices, reasons
