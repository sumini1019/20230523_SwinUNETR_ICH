# protobuf 3.20
import pandas as pd
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import glob
import time
import logging
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchio as tio
import nibabel as nib
from tensorboardX import SummaryWriter
# from models import criterions
# from pytorch_toolbelt import losses as L
from loss import MyFocalTverskyLoss, MyDiceLoss, SegMetrics_sumin
from tqdm import tqdm
from monai.losses import DiceCELoss, TverskyLoss, DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.data import DataLoader as DataLoaderMonai
from monai.data import (
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

def main():
    device_num = "0" # 사용할 GPU 번호를 설정합니다.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # GPU 장치 순서를 설정합니다.
    os.environ["CUDA_VISIBLE_DEVICES"] = device_num # 사용할 GPU를 설정합니다.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 장치를 설정합니다.

    print('Device:', device_num)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    ### Setting ###
    ranseed = 9006  # 랜덤 시드를 설정합니다.
    # Tversky Loss에서, FP / FN에 대한 가중치
    # - alpha : 높을수록, False Positive를 회피함 (민감도 감소 / 특이도 증가)
    # - beta  : 높을수록, False Negative를 회피함 (민감도 증가 / 특이도 감소)
    alpha, beta = 0.74, 0.26

    size_base = 256     #192
    size_resize = 256   #192
    size_crop = 256     #192
    learning_rate = 1e-4
    epochs = 20
    overlap_numer = 3
    overlap_denom = 4

    # Set Train Patch
    train_batch_size = 2  # 훈련 배치 크기 설정
    test_batch_size = 2  # validation 배치 크기 설정

    patch_size = 64  # 이미지 패치 크기 설정
    samples_per_volume = 64 #512  # 볼륨 당 샘플 개수 설정
    max_queue_length = 8  # 최대 큐 길이 설정

    # 2023.06.30
    # - Train 시, Result Pred의 Dice Score 계산 위한 threshold
    threshold_dice = 0.5

    # Windowing Range
    window_min = 0
    window_max = 80

    name_project = f'20230728_Try4_HemoRate{beta}_BaseSize{size_base}_PatchSize{patch_size}_LR{learning_rate}'

    # 폴더 생성
    log_dir = os.path.join(os.getcwd(), 'log', name_project)
    log_path = os.path.join(log_dir, time.strftime("%y%m%d%H%M", time.localtime()) + '.txt') # 로그 경로를 설정합니다.
    os.makedirs(log_dir, exist_ok=True) # 로그 디렉토리를 생성합니다.

    # 로그 설정
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])

    checkpoint_dir = os.path.join(os.getcwd(), 'output_models', name_project)
    os.makedirs(checkpoint_dir, exist_ok=True) # 체크포인트 디렉토리를 생성합니다.

    recon_dir = os.path.join(os.getcwd(), 'output_pred_image', name_project)
    os.makedirs(recon_dir, exist_ok=True) # 재구성 디렉토리를 생성합니다.

    tensorboard_dir = os.path.join(os.getcwd(), 'tensorboard', name_project)
    os.makedirs(tensorboard_dir, exist_ok=True)  # 재구성 디렉토리를 생성합니다.



    # 로깅을 이용해 초기 매개변수 출력
    logging.info(
        'seed{} alpha{} beta{} patch{}'
            .format(ranseed, alpha, beta, patch_size))

    # Set Path
    image_dir = r'D:\00000000_Data\20230630_ICH_SwinUNETR_Data\until_7th\image_Hemo_nifti (ALL)_Resize(256, 256, 64)'
    gt_dir = r'D:\00000000_Data\20230630_ICH_SwinUNETR_Data\until_7th\label_nrrd(until_7th)_Resize(256, 256, 64)_Binary_exclude_ChronicSDH'
    # image_dir = r'D:\00000000_Data\20230630_ICH_SwinUNETR_Data\until_7th\image_Hemo_nifti (ALL)_Resize(256, 256, 256)'
    # gt_dir = r'D:\00000000_Data\20230630_ICH_SwinUNETR_Data\until_7th\label_nrrd(until_7th)_Resize(256, 256, 256)_Binary_exclude_ChronicSDH'
    path_GT_csv = r'D:\00000000_Data\20230630_ICH_SwinUNETR_Data\until_7th\20230811_GT_ICH_Annotation_n3727_7차수령데이터까지.csv'
    df = pd.read_csv(path_GT_csv)

    train_img_path, train_gt_path = [], []
    test_img_path, test_gt_path = [], []
    # DataFrame 순회하며 Type에 따라 경로 분류
    for index, row in df.iterrows():
        fn = row['Image'].replace('.nii.gz', '')
        image_path = os.path.join(image_dir, fn + '.nii.gz')
        gt_path = os.path.join(gt_dir, fn + '-label.nii.gz')  # + '-label.nrrd'

        # 파일 경로가 유효한지 확인
        if not os.path.exists(image_path) or not os.path.exists(gt_path):
            print(f"Warning: File paths do not exist for {fn}")
            continue

        if row['Type'] == 'Test':
            test_img_path.append(image_path)
            test_gt_path.append(gt_path)
        else:
            train_img_path.append(image_path)
            train_gt_path.append(gt_path)

    # data_dir = r'D:\00000000_Data\20230630_ICH_SwinUNETR_Data'
    # train_dir = os.path.join(data_dir, 'train')
    # train_img_dir = os.path.join(train_dir, 'image_Hemo_nifti_Resize_256')
    # train_gt_dir = os.path.join(train_dir, 'label_nrrd(until_5th)_Resize_256_exclude_ChronicSDH')
    # test_dir = os.path.join(data_dir, 'test')
    # test_img_dir = os.path.join(test_dir, 'image_Hemo_nifti_Resize_256')
    # test_gt_dir = os.path.join(test_dir, 'label_nrrd(until_5th)_Resize_256_exclude_ChronicSDH')

    # # 정렬된 이미지 및 라벨 경로 리스트 생성
    # train_img_path = sorted(glob.glob(train_img_dir + '/*.nii.gz'))
    # train_gt_path = sorted(glob.glob(train_gt_dir + '/*.nrrd'))
    # test_img_path = sorted(glob.glob(test_img_dir + '/*.nii.gz'))
    # test_gt_path = sorted(glob.glob(test_gt_dir + '/*.nrrd'))

    # Slicing?
    do_slice = True
    if do_slice:
        # Set data list
        # 훈련 및 테스트 인덱스 리스트 설정
        train_indices = list(range(5))
        test_indices = list(range(2))

        # 훈련 및 테스트 인덱스를 파일로 저장
        with open(log_path + 'trainindex.txt', 'w') as f:
            f.write("\n".join([str(x) for x in train_indices]))

        with open(log_path + 'testindex.txt', 'w') as f:
            f.write("\n".join([str(x) for x in test_indices]))

        # 훈련 및 테스트 이미지와 라벨 경로 리스트에서 해당 인덱스만 선택
        train_img_path = [train_img_path[i] for i in train_indices]
        train_gt_path = [train_gt_path[i] for i in train_indices]
        test_img_path = [test_img_path[i] for i in test_indices]
        test_gt_path = [test_gt_path[i] for i in test_indices]

    '''
    test_img_path = [test_img_path[i] for i in test_indices]
    test_gt_path = [test_gt_path[i] for i in test_indices]
    '''

    # Set SubjectsDataset
    # 훈련 및 테스트 데이터셋 설정
    train_subjects = []
    for (img_path, gt_path) in zip(train_img_path, train_gt_path):
        subject = tio.Subject(
            img=tio.ScalarImage(img_path),
            gt=tio.LabelMap(gt_path),
        )
        train_subjects.append(subject)

    # 같은 방식으로 테스트 데이터셋 설정
    test_subjects = []
    for (img_path, gt_path) in zip(test_img_path, test_gt_path):
        subject = tio.Subject(
            img=tio.ScalarImage(img_path),
            gt=tio.LabelMap(gt_path),
        )
        test_subjects.append(subject)

    '''
    test_subjects = []
    for (test_img_path, test_gt_path) in zip(test_img_path, test_gt_path):
        subject = tio.Subject(
            mri=tio.ScalarImage(test_img_path),
            brain=tio.LabelMap(test_gt_path),
        )
        test_subjects.append(subject)
    test_dataset = tio.SubjectsDataset(test_subjects)
    '''

    # Set Transform
    train_transform = tio.Compose([
        # 2023.05.23
        # - train loader의 이미지가 같은 공간에 있지 않다는 에러에 대한 대응
        # tio.ToCanonical(),
        # tio.Resample((512, 512, 256)),
        # tio.Resample((512, 512, 512)),

        # tio.CropOrPad(size_base),  # 기본 크기로 이미지를 자르거나 채웁니다.

        # tio.CropOrPad((size_base, size_base, size_base)),  # 기본 크기로 이미지를 자르거나 채웁니다.

        # tio.RandomFlip(),  # 랜덤으로 이미지를 뒤집습니다.

        # tio.RandomAffine(default_pad_value=0),  # 랜덤한 어파인 변환을 적용합니다.
        # tio.RandomBiasField(),  # 랜덤한 바이어스 필드를 추가합니다.

        # tio.Resize(size_resize),  # 이미지의 크기를 재조정합니다.
        # tio.CropOrPad(size_crop),  # 이미지를 잘라내거나 크기를 조정합니다.

        # tio.RandomMotion(),  # 랜덤 모션을 추가합니다.
        # tio.RandomAnisotropy(),  # 랜덤한 이방성을 추가합니다.
        # tio.RandomElasticDeformation(),  # 랜덤한 탄성 변형을 추가합니다.
        # tio.RandomGhosting(),  # 랜덤한 유령 효과를 추가합니다.
        # tio.RandomSpike(),  # 랜덤한 스파이크를 추가합니다.
        # tio.RandomNoise(),  # 랜덤한 노이즈를 추가합니다.
        # tio.RandomBlur(),  # 랜덤한 블러 효과를 추가합니다.
        # tio.RandomSwap(),  # 랜덤하게 픽셀을 교환합니다.
        # tio.RandomGamma(),  # 랜덤한 감마 변환을 적용합니다.
        # tio.Pad((patch_size//2) + 1),  # 패딩을 추가합니다.

        # tio.CropOrPad(size_base),  # 기본 크기로 이미지를 자르거나 채웁니다.
        # tio.Resize(size_resize),  # 이미지의 크기를 재조정합니다.
        # tio.CropOrPad(size_base),  # 기본 크기로 이미지를 자르거나 채웁니다.

        tio.CropOrPad((size_base, size_base, size_base)),  # 기본 크기로 이미지를 자르거나 채웁니다.
        tio.Clamp(out_min=window_min, out_max=window_max),  # 0-80 사이로 Intensity를 재조정합니다.

        tio.RandomAffine(default_pad_value=0),  # 랜덤한 어파인 변환을 적용합니다.
        tio.RandomNoise(),  # 랜덤한 노이즈를 추가합니다.
        # tio.RandomBiasField(),  # 랜덤한 바이어스 필드를 추가합니다.
        tio.RandomGamma(log_gamma=(0.01, 0.01)),  # 랜덤한 감마 변환을 적용합니다.

    ])

    # 이전 코드와 동일하지만, 랜덤한 변환들을 제외하고, 이미지를 재조정하고 잘라내는 단계만 포함합니다.
    test_transform = tio.Compose([
        # 2023.05.23
        # - train loader의 이미지가 같은 공간에 있지 않다는 에러에 대한 대응
        # tio.ToCanonical(),
        # tio.Resample((512, 512, 36)),

        # tio.CropOrPad(size_base),
        # tio.Resize(size_resize),
        # tio.CropOrPad(size_crop),

        tio.CropOrPad((size_base, size_base, size_base)),
        tio.Clamp(out_min=0, out_max=80),  # 0-80 사이로 Intensity를 재조정합니다.
    ])

    # Set dataset
    train_set = tio.SubjectsDataset(
        train_subjects, transform=train_transform)  # 훈련 데이터셋 설정

    test_set = tio.SubjectsDataset(
        test_subjects, transform=test_transform)  # 테스트 데이터셋 설정

    visualize = False
    if visualize:
        # Get the first subject from the dataset
        subject = train_set[4]
        # Get the image and label tensors
        image_tensor = subject['img']['data']
        label_tensor = subject['gt']['data']
        image_tensor = image_tensor.squeeze()
        label_tensor = label_tensor.squeeze()
        print(image_tensor.shape)
        print(label_tensor.shape)
        # Convert the tensors to numpy arrays
        image_array = image_tensor.numpy()
        label_array = label_tensor.numpy()

        clipped_image_array = image_array

        # Plot the middle slice of the image and label
        middle_slice = clipped_image_array.shape[2] // 2
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(clipped_image_array[:, :, middle_slice], cmap='gray')
        plt.title('Image')
        plt.subplot(1, 2, 2)
        plt.imshow(label_array[:, :, middle_slice], cmap='gray')
        plt.title('Label')
        plt.show()

    # 여기서는 균일한 샘플링을 사용합니다. 주석 처리된 코드는 레이블 별 확률을 지정하여 샘플링하는 코드입니다.
    train_sampler = tio.data.UniformSampler(patch_size)
    train_sampler = tio.data.LabelSampler(patch_size, label_probabilities={0: 0.5, 1: 0.5})

    # 이 Queue 객체는 훈련 중에 훈련 패치를 효율적으로 로드하는데 사용됩니다.
    patches_train_set = tio.Queue(
        subjects_dataset=train_set,             # 환자 기준 데이터
        max_length=max_queue_length,            # patch 큐의 최대 길이 (?)
        samples_per_volume=samples_per_volume,  # 각 환자에서 추출할 패치의 수
        sampler=train_sampler,                  # 패치 사이즈가 결정되어 있는, 샘플러
        num_workers=0, #16,
        shuffle_subjects=False,
        shuffle_patches=True,
    )

    if visualize:
        for i in range(10):
            # Get the first subject from the dataset
            subject = patches_train_set[i]

            # Get the image and label tensors
            image_tensor_us = subject['img']['data']
            label_tensor_us = subject['gt']['data']

            image_tensor = image_tensor_us.squeeze()
            label_tensor = label_tensor_us.squeeze()

            # Convert the tensors to numpy arrays
            image_array = image_tensor.numpy()
            label_array = label_tensor.numpy()

            # Windowing 수행
            window_min = 0
            window_max = 80
            clipped_image_array = np.clip(image_array, window_min, window_max)

            # Plot the middle slice of the image and label
            middle_slice = image_array.shape[2] // 2

            if label_array[:, :, middle_slice].max() > 0:
                print(label_array[:, :, middle_slice].max())

                plt.figure(figsize=(12, 6))

                plt.subplot(1, 2, 1)
                plt.imshow(clipped_image_array[:, :, middle_slice], cmap='gray', vmin=0, vmax=40)
                plt.title('Image')

                plt.subplot(1, 2, 2)
                plt.imshow(label_array[:, :, middle_slice], cmap='gray', vmin=0, vmax=1)
                plt.title('Label')

                plt.show()




    # 이 DataLoader 객체는 훈련 중에 훈련 패치를 효율적으로 로드하는데 사용됩니다.
    train_loader_patches = torch.utils.data.DataLoader(
        patches_train_set, batch_size=train_batch_size)

    # Set Model
    model = SwinUNETR(
        in_channels=1,  # in_channels=1,
        out_channels=2,  # out_channels=2,
        img_size=(patch_size, patch_size, patch_size),  # img_size=(256, 256, 256),
        depths=(4, 4, 4, 4),  # depths=(2,2,2,2),
        num_heads=(3, 6, 12, 24),  # num_heads=(3,6,12,24),
        feature_size=48,  # feature_size=24,
        # feature_size=16,
        # hidden_size=768,
        # mlp_dim=3072,
        # pos_embed="perceptron",
        norm_name="instance",
        # res_block=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
    ).to(device)

    # 손실 함수 및 최적화 알고리즘을 설정합니다.
    criterion_a = MyFocalTverskyLoss(alpha=0.7, gamma=2.3)
    criterion_b = SegMetrics_sumin()


    loss_function = TverskyLoss(to_onehot_y=True, softmax=False, alpha=alpha, beta=beta, smooth_dr=1e-6, smooth_nr=1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # 학습률 스케줄러 설정
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.95 ** epoch)

    # 스케일러는 부동소수점 연산의 오버플로우와 언더플로우를 관리합니다.
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # 로그를 작성하기 위한 writer를 생성합니다.
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # 현재 시간을 기록하여 훈련 시간을 측정합니다.
    start_time = time.time()

    # 초기 최적 손실 값을 설정합니다.
    epoch_loss_best = [0, 1.0]  # [epoch, value_best]
    epoch_dice_best = [0, 1.0]  # [epoch, value_best]

    # 여기서는 훈련과 검증을 실행하는 메인 루프가 시작됩니다.
    for epoch in range(epochs):

        time_startepoch = time.time()

        model.train()
        total_TverskyLoss_Train = 0
        total_DiceLoss_Train = 0
        total_Accuracy_Train = 0
        total_Sensitivity_Train = 0
        total_Specificity_Train = 0
        epoch_numbest = 0

        # 2023.07.21
        # - tqbm 추가
        train_loader_patches = tqdm(train_loader_patches)

        # 배치마다 데이터를 로드하고, 모델을 통해 예측을 수행하고, 손실을 계산하고, 가중치를 업데이트합니다.
        for i, data in enumerate(train_loader_patches):
            with torch.cuda.amp.autocast():
                x_image, y_label = data['img']['data'].to(device), data['gt']['data'].to(device)

                # 2023.06.30
                # - Image에 대한 Windowing 추가

                # output = model(x)
                output = model(x_image.float())

                output = torch.softmax(output, axis=1)
                # output = torch.sigmoid(output, axis=1)

                y_label = 1 * (y_label > 0)       # Label 이진화 (0보다 크면 1로)
                loss = loss_function(output, y_label)
                output_thr = 1. * (output > threshold_dice)        # 모델 결과를 이진화
                dice, patch_Accuracy, patch_Sensitivity, patch_Specificity = criterion_b(y_true=y_label, y_pred=output_thr)
                # criterion_c = MyDiceLoss()
                # dice2, accuracy2, sensitivity2, specificity2 = criterion_c(y_true=target, y_pred=output_thr)

                visualize = False
                if visualize:
                    image_array = x_image.detach().cpu().numpy()[0, 0, :, :, :]
                    label_array = y_label.detach().cpu().numpy()[0, 0, :, :, :]
                    output_array_BG = output.detach().cpu().numpy()[0, 0, :, :, :]
                    output_array_Hemo = output.detach().cpu().numpy()[0, 1, :, :, :]
                    output_array_thr_BG = output_thr.detach().cpu().numpy()[0, 0, :, :, :]
                    output_array_thr_Hemo = output_thr.detach().cpu().numpy()[0, 1, :, :, :]

                    # Plot the middle slice of the image and label
                    middle_slice = image_array.shape[2] // 2

                    if label_array[:, :, middle_slice].max() > 0:
                        plt.figure(figsize=(12, 12))

                        plt.subplot(3, 2, 1)
                        plt.imshow(image_array[:, :, middle_slice], cmap='gray', vmin=0, vmax=40)
                        plt.title('Image')

                        plt.subplot(3, 2, 2)
                        plt.imshow(label_array[:, :, middle_slice], cmap='gray', vmin=0, vmax=1)
                        plt.title('Label')

                        plt.subplot(3, 2, 3)
                        plt.imshow(output_array_BG[:, :, middle_slice], cmap='gray', vmin=0, vmax=1)
                        plt.title('output_BG')

                        plt.subplot(3, 2, 4)
                        plt.imshow(output_array_Hemo[:, :, middle_slice], cmap='gray', vmin=0, vmax=1)
                        plt.title('output_Hemo')

                        plt.subplot(3, 2, 5)
                        plt.imshow(output_array_thr_BG[:, :, middle_slice], cmap='gray', vmin=0, vmax=1)
                        plt.title('output_BG_Binary')

                        plt.subplot(3, 2, 6)
                        plt.imshow(output_array_thr_Hemo[:, :, middle_slice], cmap='gray', vmin=0, vmax=1)
                        plt.title('output_Hemo_Binary')

                        plt.tight_layout()
                        plt.show()

            patch_TverskyLoss = loss.data.cpu().numpy()
            patch_DiceLoss = dice.data.cpu().numpy()

            # Loss, Metric 누적
            total_TverskyLoss_Train += patch_TverskyLoss
            total_DiceLoss_Train += patch_DiceLoss
            total_Accuracy_Train += patch_Accuracy.cpu().numpy()
            total_Sensitivity_Train += patch_Sensitivity.cpu().numpy()
            total_Specificity_Train += patch_Specificity.cpu().numpy()

            # epoch의 평균 Loss, Metric 계산
            epoch_TverskyLoss_Train = total_TverskyLoss_Train / (i + 1)
            epoch_DiceLoss_Train = total_DiceLoss_Train / (i + 1)
            epoch_Accuracy_Train = total_Accuracy_Train / (i + 1)
            epoch_Sensitivity_Train = total_Sensitivity_Train / (i + 1)
            epoch_Specificity_Train = total_Specificity_Train / (i + 1)

            # 가중치 업데이트
            optimizer.zero_grad()
            scaler.scale(loss + dice).backward()
            scaler.step(optimizer)
            scaler.update()

            # 전체 훈련 과정에서의 배치 처리 횟수 계산
            global_step_Train = epoch * len(train_loader_patches) + i

            # logging.info(
            #     f'Epoch:{epoch}_Iter:{i} || tversky: {avg_TverskyLoss:.4f} || dice: {avg_DiceLoss:.4f} || '
            #     f'ACC: {avg_Accuracy:.4f} || TPR: {avg_Sensitivity:.4f} || TNR: {avg_Specificity:.4f} || TRAIN')
            train_loader_patches.set_description(
                    f"[Epoch {epoch}/{epochs} Iter {i}] TverskyLoss: {epoch_TverskyLoss_Train:.4f} DiceLoss: {epoch_DiceLoss_Train:.4f}"
                    f" ACC: {epoch_Accuracy_Train:.4f} TPR: {epoch_Sensitivity_Train:.4f} TNR: {epoch_Specificity_Train:.4f}")

            # Tensorboard 로그 기록
            writer.add_scalar('train_lr', optimizer.param_groups[0]['lr'], global_step_Train)
            writer.add_scalar('train_TverskyLoss_PatchWise', epoch_TverskyLoss_Train, global_step_Train)
            writer.add_scalar('train_DiceLoss_PatchWise', epoch_DiceLoss_Train, global_step_Train)
            writer.add_scalar('train_Acc_PatchWise', epoch_Accuracy_Train, global_step_Train)
            writer.add_scalar('train_TPR_PatchWise', epoch_Sensitivity_Train, global_step_Train)
            writer.add_scalar('train_TNR_PatchWise', epoch_Specificity_Train, global_step_Train)
            # writer.add_scalar('train_lr', optimizer.param_groups[0]['lr'], epoch)
            # writer.add_scalar('train_TverskyLoss_PatchWise', epoch_TverskyLoss / i, epoch)
            # writer.add_scalar('train_DiceLoss_PatchWise', epoch_DiceLoss, epoch)
            # writer.add_scalar('train_Acc_PatchWise', epoch_Accuracy)
            # writer.add_scalar('train_TPR_PatchWise', epoch_Sensitivity)
            # writer.add_scalar('train_TNR_PatchWise', epoch_Specificity)

        # At the end of the epoch, you close the tqdm progress bar
        train_loader_patches.close()

        # 스케줄러 단계 수행
        scheduler.step()
        time_endepoch = time.time()

        # 모델 검증 단계
        # - 해당 Validation 단계에서는,
        # - 환자 기준으로 Loss를 계산함
        # - 1. 환자 데이터를 Grid로 나눠서 patch 단위로 예측 후 aggregate
        # - 2. 환자 기준으로 합성된 예측 결과를 기반으로 loss 및 성능 metric 계산
        model.eval()
        with torch.no_grad():
            total_TverskyLoss_Val = 0
            total_DiceLoss_Val = 0
            total_Accuracy_Val = 0
            total_Sensitivity_Val = 0
            total_Specificity_Val = 0

            # 모든 테스트 셋을 순회하면서 GridSampler를 사용해 테스트 패치를 생성하고, 패치를 모델에 통과시키며, 결과를 GridAggregator에 추가합니다.
            test_loader_patients = tqdm(test_set, desc=f'Validation Epoch {epoch}')
            for i, data in enumerate(test_loader_patients):
                ########################################################################################
                # 환자를 Grid로 나눠서, patch 단위로 예측하고, 다시 모으는 과정
                # - GridSampler와 GridAggregator를 사용하여 큰 3D 볼륨을 여러 개의 작은 패치로 분할
                # - 각 패치를 모델에 통과시킨 후 결과를 다시 합쳐서 전체 볼륨의 예측을 생성하는 과정 진행
                # - 이는 큰 볼륨을 한 번에 처리하는 것이 어려운 경우에 유용하게 사용될 수 있습니다.

                # 1. GridSampler 생성
                # - GridSampler는 전체 볼륨을 겹치는 패치로 분할하는 데 사용
                # - patch_size는 각 패치의 크기를 정의하며, patch_overlap는 서로 다른 패치 간의 겹치는 부분의 크기를 정의
                grid_sampler = tio.inference.GridSampler(data, patch_size,
                                                         patch_overlap=(patch_size * overlap_numer) // overlap_denom)

                # 2. DataLoader 생성
                # DataLoader는 패치를 배치로 묶는 데 사용됩니다.
                # batch_size=2는 각 배치에 두 개의 패치가 포함되도록 설정합니다.
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=test_batch_size)

                # 3. GridAggregator 생성
                # GridAggregator는 각 패치에 대한 모델의 예측 결과를 모아서 전체 볼륨의 예측을 생성하는 데 사용
                aggregator = tio.inference.GridAggregator(grid_sampler)

                # 4. 각 배치에 대해 모델을 실행하고 결과를 aggregator에 추가
                # - 각 패치 배치에 대해, 패치를 모델에 통과시켜 예측 결과를 얻고,
                # - 그런 다음, 이 예측 결과와 해당 패치의 위치를 GridAggregator에 추가
                for patches_batch in patch_loader:
                    input_tensor = patches_batch['img'][tio.DATA].to(device)
                    locations = patches_batch[tio.LOCATION]
                    outputs = model.forward(input_tensor.float())

                    aggregator.add_batch(outputs, locations)

                # 5. GridAggregator에서 출력 텐서를 얻기
                # - 모든 패치에 대한 예측이 완료되면, GridAggregator를 사용하여 이들 예측을 모아 전체 볼륨의 예측을 생성
                output_tensor = aggregator.get_output_tensor()
                ########################################################################################





                # 소프트맥스를 적용하고 차원을 추가합니다.
                # - output_tensor에 대해 softmax를 적용하여 확률로 변환하고, 배치 차원을 추가
                # - Softmax는 모델의 출력을 확률로 해석할 수 있도록 돕습니다.
                # - torch.unsqueeze은 추가적인 차원을 생성하는데, 이는 다음 단계에서 라벨과 함께 처리될 수 있도록 합니다.
                output_tensor = torch.unsqueeze(torch.softmax(output_tensor, axis=0), dim=0)

                # 대상 텐서에 차원을 추가합니다.
                # - Ground truth 라벨 텐서에 대해 차원을 추가합니다. 이는 모델의 출력과 동일한 차원을 가지도록 하기 위함
                y_label = torch.unsqueeze(data['gt']['data'], dim=0)

                # 출력 텐서 임계치 적용 및 대상 텐서 이진화
                output_tensor_thr = 1. * (output_tensor > 0.5)
                y_label = 1 * (y_label > 0)

                # loss 계산
                loss = loss_function(output_tensor_thr, y_label)

                # 다이스 계수를 계산합니다.
                dice, patient_Accuracy_Val, patient_Sensitivity_Val, patient_Specificity_Val = criterion_b(y_true=y_label, y_pred=output_tensor_thr)

                # 손실과 다이스 계수를 CPU로 이동하고 numpy 배열로 변환합니다.
                patient_TverskyLoss_Val = loss.data.cpu().numpy()
                patient_DiceLoss_Val = dice.data.cpu().numpy()

                # Loss, Metric 누적
                total_TverskyLoss_Val += patient_TverskyLoss_Val
                total_DiceLoss_Val += patient_DiceLoss_Val
                total_Accuracy_Val += patient_Accuracy_Val
                total_Sensitivity_Val += patient_Sensitivity_Val
                total_Specificity_Val += patient_Specificity_Val

                # 평균 Loss, Metric 계산
                epoch_TverskyLoss_Val = total_TverskyLoss_Val / (i + 1)
                epoch_DiceLoss_Val = total_DiceLoss_Val / (i + 1)
                epoch_Accuracy_Val = total_Accuracy_Val / (i + 1)
                epoch_Sensitivity_Val = total_Sensitivity_Val / (i + 1)
                epoch_Specificity_Val = total_Specificity_Val / (i + 1)

                test_loader_patients.set_description(
                    f"[Epoch {epoch}/{epochs} Iter {i}] TverskyLoss: {epoch_TverskyLoss_Val:.4f} DiceLoss: {epoch_DiceLoss_Val:.4f} "
                    f"ACC: {epoch_Accuracy_Val:.4f} TPR: {epoch_Sensitivity_Val:.4f} TNR: {epoch_Specificity_Val:.4f} "
                    f"Test fname: {os.path.basename(test_gt_path[i])}")

                # 전체 validation 과정에서의 배치 처리 횟수 계산
                global_step_Val = epoch * len(test_loader_patients) + i

                # Tensorboard 로그 기록
                writer.add_scalar('val_Loss_Tversky_PatchWise', epoch_TverskyLoss_Val, global_step_Val)
                writer.add_scalar('val_Loss_Dice_PatchWise', epoch_DiceLoss_Val, global_step_Val)
                writer.add_scalar('val_Acc_PatientWise', epoch_Accuracy_Val, global_step_Val)
                writer.add_scalar('val_TPR_PatientWise', epoch_Sensitivity_Val, global_step_Val)
                writer.add_scalar('val_TNR_PatientWise', epoch_Specificity_Val, global_step_Val)


                # 2023.06.28
                # - 결과 이미지 저장
                # - Image를 Recon 해서 만드는 형식으로 변경
                # - 기존 코드는, 라벨 파일을 저장하려고 했던거 같은데... 오류 남
                sname = test_gt_path[i].replace('-label', '').split('\\')[-1].split('.')[0]
                orig_dir = image_dir #test_img_dir
                orig_fpath = os.path.join(orig_dir, sname + '.nii.gz')
                orig_nii = nib.load(orig_fpath)

                orig_img = orig_nii.get_fdata()
                orig_size = orig_img.shape

                subj = tio.Subject(gt=tio.LabelMap(tensor=output_tensor[0, :, :, :, :].cpu().numpy()))
                subj = test_transform(subj)

                # 2023.07.14
                # - Hemorrhage Pred 만 저장
                recon_img = subj['gt']['data'][1].cpu().numpy()
                recon_nii = nib.Nifti1Image(recon_img, orig_nii.affine, orig_nii.header)

                recon_fpath = os.path.join(recon_dir, sname + f'_pred_ep{epoch}.nii.gz')
                nib.save(recon_nii, recon_fpath)

            # # 테스트 세트의 크기를 계산합니다.
            # num_test = len(test_set)
            # # 에폭에 대한 평균 손실과 다이스 계수를 계산합니다.
            # epoch_loss_current = total_TverskyLoss / num_test
            # epoch_dice_current = total_DiceLoss / num_test
            #
            # # 선택된 배치에 대한 평균 손실과 다이스 계수를 계산합니다.
            # select_loss_current = select_loss / num_test
            # select_dice_current = select_dice / num_test


            epoch_loss_best = [epoch, epoch_TverskyLoss_Val]
            epoch_dice_best = [epoch, epoch_DiceLoss_Val]

            file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            }, file_name)

                # # 이전 체크포인트를 삭제합니다.
                # ckpt_rm_list = [x for x in sorted(glob.glob(os.path.join(checkpoint_dir, '*.pth'))) if
                #                 '_{}.pth'.format(epoch) not in x]
                # for ckpt_rm in ckpt_rm_list:
                #     os.remove(ckpt_rm)

            # 로깅 메시지를 생성합니다.
            logging.info(
                f'Epoch:{epoch}_NumtestBatch:{len(test_set)} '
                f'tversky: {epoch_TverskyLoss_Val:.4f} dice: {epoch_DiceLoss_Val:.4f} '
                f'- best (Tversky): {epoch_loss_best[0]},{epoch_loss_best[1]:.4f}'
                f'- best (Dice): {epoch_dice_best[0]},{epoch_dice_best[1]:.4f}')

        logging.info(
            '----------------------------------The testation process finished!-----------------------------------')

        # 에폭별 시간 소모와 예상 남은 시간을 계산하고 로깅합니다.
        epoch_time_minute = (time_endepoch - time_startepoch) / 60
        remaining_time_hour = (epochs - epoch - 1) * epoch_time_minute / 60
        logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
        logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))

    writer.close()

    # 마지막 모델 저장
    final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
    torch.save({
        'epoch': epochs,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
    }, final_name)

    # 전체 훈련 시간 확인
    end_time = time.time()
    total_time = (end_time - start_time) / 3600
    logging.info('The total training ztime is {:.2f} hours'.format(total_time))

    logging.info('----------------------------------The training process finished!-----------------------------------')

if __name__ == '__main__':
    main()