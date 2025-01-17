import os


class GlobalConfig:
    ctrl_opt = 'one_of'  # one_of both_must pid_only mlp_only
    gpu_id = '0'
    # model = 'xr14'
    # model = 'vit_bb'
    model = 'eff_vit'
    logdir = 'log/'+model+'_mix_mix'
    init_stop_counter = 30

    batch_size = 8
    coverage_area = 24  # untuk top view SC, 24m kedepan, kiri, dan kanan
    rp1_close = 4  # ganti rp jika mendekati ...meter
    bearing_bias = 7.5  # dalam derajat, pastikan sama dengan yang ada di plot_wprp.py
    n_buffer = 0  # buffer untuk MAF dalam second
    data_rate = 4  # 1 detik ada berapa data?

    # parameter untuk MGN
    MGN = False
    loss_weights = [1, 1, 1, 1]
    lw_alpha = 1.5
    bottleneck = [335, 675]  # cek dengan check_arch.py

    # Data
    seq_len = 1  # jumlah input seq
    pred_len = 3  # future waypoints predicted
    logdir = logdir+"_seq"+str(seq_len)  # update direktori name

    # root_dir = '/home/aisl/WHILL/ros-whill-robot/main/dataset'
    root_dir = './dataset/dataset_0'
    train_dir = root_dir+'/train_routes'
    val_dir = root_dir+'/val_routes'
    test_dir = root_dir+'/test_routes'
    # train: sunny0,2,4,6,8,11 sunset1,3,5,7,9,10
    # sunny route 2,4,6,11 ada sedikit adversarial
    train_conditions = ['sunny']
    # val: sunny1,3,5,7,9,10 sunset0,2,4,6,8,11
    val_conditions = ['sunny']  # pokoknya kebalikannya train
    test_conditions = ['sunny']
    # train_data, val_data, test_data = [], [], []
    # for weather in weathers:
    #     train_data.append(os.path.join(root_dir+'/train_routes', weather))
    #     val_data.append(os.path.join(root_dir+'/val_routes', weather))
    # test_weathers = ['cloudy']
    # for weather in test_weathers:
    #     test_data.append(os.path.join(root_dir+'/test_routes', weather))

    crop_roi = [512, 1024]  # HxW
    scale = 1  # buat resizinig diawal load data
    # res_resize = [256, 384]
    res_resize = [224, 224]
    # res_resize = [256, 256]
    # res_resize = [512, 768]
    # res_resize = [256, 512]

    lr = 1e-4  # learning rate #pakai AdamW
    weight_decay = 1e-3

    # Controller
    # control weights untuk PID dan MLP dari tuningan MGN
    # urutan steering, throttle
    # baca dulu trainval_log.csv setelah training selesai, dan normalize bobotnya 0-1
    # LWS: lw_wp lw_str lw_thr saat convergence
    lws = [1, 1, 1]
    cw_pid = [lws[0]/(lws[0]+lws[1]), lws[0]/(lws[0]+lws[2])]  # str, thrt
    cw_mlp = [1-cw_pid[0], 1-cw_pid[1]]  # str, thrt, brk

    turn_KP = 0.5
    turn_KI = 0.25
    turn_KD = 0.15
    turn_n = 15  # buffer size

    speed_KP = 1.5
    speed_KI = 0.25
    speed_KD = 0.5
    speed_n = 15  # buffer size

    max_throttle = 1.0  # upper limit on throttle signal value in dataset
    wheel_radius = 0.15  # radius roda robot dalam meter
    # brake_speed = 0.4 # desired speed below which brake is triggered
    # brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    # clip_delta = 0.25 # maximum change in speed input to logitudinal controller
    min_act_thrt = 0.1  # minimum nilai suatu throttle dianggap aktif diinjak
    err_angle_mul = 0.075
    des_speed_mul = 1.75

    # BACA https://mmsegmentation.readthedocs.io/en/latest/_modules/mmseg/core/evaluation/class_names.html#get_palette
    # HANYA ADA 19 CLASS?? + #tambahan 0,0,0 hitam untuk area kosong pada SDC nantinya
    SEG_CLASSES = {
        'colors': [[0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                   [190, 153, 153], [153, 153, 153], [
            250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [
            70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
            [0, 80, 100], [0, 0, 230], [119, 11, 32]],
        'classes': ['not_class', 'road', 'sidewalk', 'building', 'wall',
                    'fence', 'pole', 'traffic light', 'traffic sign',
                    'vegetation', 'terrain', 'sky', 'person',
                    'rider', 'car', 'truck', 'bus',
                    'train', 'motorcycle', 'bicycle']
    }
    n_class = len(SEG_CLASSES['colors'])

    n_fmap_b0 = [[32, 16], [24], [40], [80, 112], [192, 320, 1280]]
    n_fmap_b1 = [[32, 16], [24], [40], [80, 112],
                 [192, 320, 1280]]  # sama dengan b0
    n_fmap_b2 = [[32, 16], [24], [48], [88, 120], [208, 352, 1408]]
    n_vit_b16 = [[32, 24], [32], [64], [88, 128], [208, 352, 768]]
    # lihat underdevelopment/efficientnet.py
    n_fmap_b3 = [[40, 24], [32], [48], [96, 136], [232, 384, 1536]]
    n_fmap_b4 = [[48, 24], [32], [56], [112, 160], [272, 448, 1792]]
    n_decoder = n_vit_b16
    # jangan lupa untuk mengganti model torchvision di init model.py

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
