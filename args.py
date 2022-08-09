class args():
    traindata_dir = '/home/hdr/Data/train64'
    testdata_dir = '/media/hdr/Elements SE/Dataset/test64'
    evaldata_dir = '/media/hdr/Elements SE/Dataset/eval64'
    oritestdata_dir = '/media/hdr/Elements SE/Dataset/originTestGF2128'
    sample_dir = './sample'
    checkpoint_dir = './checkpoint'
    checkpoint_backup_dir = './checkpoint/backup'
    record_dir = './log/record'
    log_dir = './log'
    model2_path = './checkpoint/edge_enhance_multi.pth'
    output_dir = './output'
    edge_enhance_multi_pretrain_model = './checkpoint/edge_enhance_multi.pth'

    max_value = 1023  # QB是2047
    epochs = 100
    lr = 0.0005  # learning rate
    batch_size = 8
    lr_decay_freq = 40
    model_backup_freq = 30
    eval_freq = 10

    data_augmentation = False

    cuda = 1    # use GPU 1