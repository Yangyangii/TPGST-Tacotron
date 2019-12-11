
class ConfigArgs:
    """
    Setting Configuration Arguments
    See the comments in this code

    """
    data_path = '../../data/kss'
    mel_dir, mag_dir = 'mels', 'mags'
    meta = 'transcript.v.1.3.txt'
    testset = 'ko_sents.txt'
    logdir = 'logs' # log directory
    sampledir = 'samples' # directory where samples are located
    mem_mode = False # load all of the mel spectrograms into memory
    log_mode = True # whether it logs
    log_term = 300 # log every n-th step
    eval_term = 1000 # log every n-th step
    synth_wav = False # whether it synthesizes waveform
    save_term = 2000 # save every n-th step
    n_workers = 4 # number of subprocesses to use for data loading
    global_step = 0 # global step

    tp_start = 100000
    sr = 22050 # sampling rate
    n_fft = 1024 # n point Fourier transform
    n_mags = n_fft//2 + 1 # magnitude spectrogram frequency
    n_mels = 80 # mel spectrogram dimension
    hop_length = 256 # hop length as a number of frames
    win_length = 1024 # window length as a number of frames
    r = 5  # reduction factor.

    batch_size = 32 # for training
    test_batch = 16 # for test
    max_step = 400000 # maximum training step
    lr = 0.001 # learning rate
    warm_up_steps = 4000.0 # warm up learning rate
    # lr_decay_step = 50000 # actually not decayed per this step
    # lr_step = [100000, 300000] # multiply 1/10
    Ce = 256  # dimension for character embedding
    Cx = 128 # dimension for context encoding
    Ca = 256 # attention dimension
    drop_rate = 0.05 # dropout rate
    n_tokens = 10 # number of tokens for style token layer
    n_heads = 8 # for multihead attention

    max_Tx = 188 # maximum length of text
    max_Ty = 250 # maximum length of audio

    JAMO_LEADS = ''.join([chr(_) for _ in range(0x1100, 0x1113)])
    JAMO_VOWELS = ''.join([chr(_) for _ in range(0x1161, 0x1176)])
    JAMO_TAILS = ''.join([chr(_) for _ in range(0x11A8, 0x11C3)])
    vocab = 'PE !,.?' + JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS
