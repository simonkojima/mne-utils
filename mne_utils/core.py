import numpy as np
import scipy
import mne


def concatenate_epochs(epochs_list, add_offset_event_id=True, **kwargs):
    if add_offset_event_id:
        events = epochs_list[0].events
        ids = events[:, 2]
        offset = np.max(ids)

        for epochs in epochs_list[1 : len(epochs_list)]:
            epochs.events[:, 2] += offset
            epochs.event_id = {k: v + offset for k, v in epochs.event_id.items()}
            offset = np.max(epochs.events[:, 2])

    return mne.concatenate_epochs(epochs_list, **kwargs)


def markers_from_events(events, event_id):
    event_desc = {v: k for k, v in event_id.items()}

    samples = np.array(events)[:, 0]

    markers = list()
    for val in np.array(events)[:, 2]:
        markers.append(str(event_desc[val]))

    return samples, markers


def events_from_markers(samples, markers, offset_id=0, event_id=None):
    unique_markers = np.unique(markers)

    if event_id is None:
        event_id = dict()
    events = list()
    for marker, sample in zip(markers, samples):
        if event_id is None:
            id = np.argwhere(unique_markers == marker)[0][0] + 1 + offset_id
        else:
            id = event_id[marker]
        events.append([sample, 0, id])
        event_id[marker] = id
    events = np.array(events)

    return events, event_id


def concatenate_tfrs(tfrs_list, add_offset=True):
    info = tfrs_list[0].info
    event_id_mutual = tfrs_list[0].event_id
    times = tfrs_list[0].times
    freqs = tfrs_list[0].freqs
    method = tfrs_list[0].method

    if add_offset:
        offset = int(np.diff(tfrs_list[0].events[:, 0]).mean())

    events_list = []
    data_list = []
    for idx, tfrs in enumerate(tfrs_list):
        events = tfrs.events
        event_id = tfrs.event_id

        samples, markers = markers_from_events(events, event_id)
        events, event_id = events_from_markers(
            samples, markers, event_id=event_id_mutual
        )

        if add_offset:
            if idx > 0:
                offset += tfrs_list[idx - 1].events[-1, 0] + offset

            events[:, 0] += offset

        events_list.append(events)
        data_list.append(tfrs.get_data())

    data = np.concatenate(data_list)
    events = np.concatenate(events_list)

    tfrs = mne.time_frequency.EpochsTFRArray(
        info=info,
        data=data,
        times=times,
        freqs=freqs,
        events=events,
        event_id=event_id_mutual,
        method=method,
    )

    return tfrs


def get_drop_indices_tfrs(tfrs, tmin=0.5, tmax=7.5, fmin=8, fmax=30, threshold=3):
    data = tfrs.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
    # std = data.std()
    # th = std * th_factor
    indices = np.where((np.abs(data) > threshold).any(axis=(1, 2, 3)))[0]

    return indices


def round_edge(x, fs, len_transition):
    """
    Parameters
    ==========
    x : data,
        shape of (n_ch, n_samples)
    len_transition : float
        length of rise/fall in seconds. This value will be used for both rise and fall.
    """

    length = x.size / fs
    alpha = len_transition / length * 2
    w = scipy.signal.windows.tukey(M=x.size, alpha=alpha)

    return x * w


def reconstruct_raw(raw):
    raw = mne.io.RawArray(
        raw.get_data(), mne.create_info(raw.ch_names, raw.info["sfreq"])
    )
    return raw


def concatenate_raws(raws, len_transition=0.5, delete_annot=True, reconstruct=False):
    cat_raws = list()
    for idx, raw in enumerate(raws):
        raw = raw.copy()

        fs = raw.info["sfreq"]
        raw.apply_function(
            round_edge,
            picks="all",
            n_jobs=-1,
            channel_wise=True,
            fs=fs,
            len_transition=len_transition,
        )

        cat_raws.append(raw)

    cat_raw = mne.concatenate_raws(cat_raws)

    N_raws = [raw.get_data().shape[1] for raw in raws]

    if np.sum(N_raws) != cat_raw.get_data().shape[1]:
        raise RuntimeError(
            "Number of samples are not same. np.sum(N_raws) != cat_raw.get_data().shape[1]"
        )

    if delete_annot:
        cat_raw.annotations.delete(list(range(len(cat_raw.annotations))))
        if len(cat_raw.annotations) != 0:
            raise RuntimeError("Some of annotations could not be deleted.")

    if reconstruct:
        return reconstruct_raw(cat_raw)
    else:
        return cat_raw


def find_bad_eog(raw, ica, threshold="max"):
    """
    Parameters
    ==========

    raw : raw instance contains eog channels.
    ica : ica instance
    filter : filter range will be used for eog channels
    threshold, numerical or 'max':

    """
    # raw_eeg_ = mne.io.RawArray(data = raw.get_data(), info = mne.create_info(raw_eeg.ch_names, Fs))

    # Fs = raw.info['sfreq']

    raw_eog = raw.copy().pick(picks="eog")
    raw_eeg = raw.copy().pick(picks="eeg")

    # reconstruct is needed somehow...
    # otherwise, the raw data will be cropped
    # maybe, it's because the raw file doesn't have info regarding concatenating
    raw_eog = reconstruct_raw(raw_eog)
    raw_eeg = reconstruct_raw(raw_eeg)

    IC = ica.get_sources(raw_eeg)

    scores = list()
    indices = list()
    for ch in raw_eog.ch_names:
        data_eog = raw_eog.get_data(picks=ch)

        score = list()
        for idx, ic in enumerate(IC.ch_names):
            a = scipy.stats.pearsonr(
                x=np.squeeze(data_eog), y=np.squeeze(IC.get_data(picks=ic))
            )
            score.append(a[0])

        if threshold == "max":
            I = np.argmax(np.absolute(np.array(score)))
            indices.append(I)
        else:
            I = np.where(np.absolute(np.array(score)) >= threshold)
            indices += I[0].tolist()

        scores.append(score)

    scores = np.array(scores)

    return indices, scores
