import matplotlib.pyplot as plt


def plot_mic_signals(provider):

    mic_stream_id = provider.get_stream_id_from_label("mic")
    num_mic_samples = provider.get_num_data(mic_stream_id)
    timestamps = []
    audio = [[] for c in range(0, 7)]
    for index in range(0, num_mic_samples):
        audio_data_i = provider.get_audio_data_by_index(mic_stream_id, index)
        audio_signal_block = audio_data_i[0].data
        timestamps_block = [t * 1e-9 for t in audio_data_i[1].capture_timestamps_ns]
        timestamps += timestamps_block
        for c in range(0, 7):
            audio[c] += audio_signal_block[c::7]

    fig, axes = plt.subplots(1, 1, figsize=(12, 5))
    fig.suptitle(f"Microphone signal")
    for c in range(0, 7):
        plt.plot(timestamps, audio[c], "-", label=f"channel {c}")
    axes.legend(loc="upper left")
    axes.grid("on")
    axes.set_xlabel("timestamps (s)")
    axes.set_ylabel("audio readout")
    plt.show()
