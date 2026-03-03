import numpy as np
import mne

def concatenate_epochs(epochs_list, add_offset_event_id = True, **kwargs):
    if add_offset_event_id:
        events = epochs_list[0].events
        ids = events[:, 2]
        offset = np.max(ids)

        for epochs in epochs_list[1:len(epochs_list)]:
            epochs.events[:,2] += offset
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


def events_from_markers(samples, markers, offset=0):
    unique_markers = np.unique(markers)

    event_id = dict()
    events = list()
    for marker, sample in zip(markers, samples):
        id = np.argwhere(unique_markers == marker)[0][0] + 1 + offset
        events.append([sample, 0, id])
        event_id[marker] = id
    events = np.array(events)

    return events, event_id