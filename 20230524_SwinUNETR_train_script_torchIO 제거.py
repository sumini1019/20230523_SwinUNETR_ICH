# protobuf 3.20
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
from loss import MyFocalTverskyLoss, MyDiceLoss
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

    ranseed = 9006 # 랜덤 시드를 설정합니다.
    alpha, beta = 0.74, 0.26 # 손실 함수 계산에 사용될 alpha, beta 값을 설정합니다.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 장치를 설정합니다.

    print('Device:', device_num)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    log_dir = 'log_traintest_gpu' + device_num # 로그 디렉토리를 설정합니다.
    log_path = os.path.join(log_dir, time.strftime("%y%m%d%H%M", time.localtime()) + '_' + str(ranseed) + '.txt') # 로그 경로를 설정합니다.
    os.makedirs(log_dir, exist_ok=True) # 로그 디렉토리를 생성합니다.

    # 로그 설정
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])

    checkpoint_dir = os.path.join(os.getcwd(), 'models_ckpt_proc_traintest_gpu' + device_num + '_' + str(ranseed) + '_' + str(alpha))
    os.makedirs(checkpoint_dir, exist_ok=True) # 체크포인트 디렉토리를 생성합니다.

    recon_dir = 'data_recon_proc_traintest_gpu' + device_num + '_' + str(ranseed) + '_' + str(alpha)
    os.makedirs(recon_dir, exist_ok=True) # 재구성 디렉토리를 생성합니다.

    size_base = 192
    size_resize = 192
    size_crop = 192
    learning_rate = 1e-4
    batch_size = 1
    epochs = 20
    overlap_numer = 3
    overlap_denom = 4

    # Set Train Patch
    train_batch_size = 2  # 훈련 배치 크기 설정
    patch_size = 64  # 이미지 패치 크기 설정
    samples_per_volume = 64 #512  # 볼륨 당 샘플 개수 설정
    max_queue_length = 4 #8  # 최대 큐 길이 설정

    # 로깅을 이용해 초기 매개변수 출력
    logging.info(
        'seed{} alpha{} beta{} patch{}'
            .format(ranseed, alpha, beta, patch_size))

    # Set Path
    # 데이터 저장 경로 설정
    data_dir = r'./data_ICH'
    train_dir = os.path.join(data_dir, 'train')
    train_img_dir = os.path.join(train_dir, 'image_Hemo_nifti')
    train_gt_dir = os.path.join(train_dir, 'label_nrrd(until_5th)')
    test_dir = os.path.join(data_dir, 'test')
    test_img_dir = os.path.join(test_dir, 'image_Hemo_nifti')
    test_gt_dir = os.path.join(test_dir, 'label_nrrd(until_5th)')

    # Set data list
    # 훈련 및 테스트 인덱스 리스트 설정
    train_indices = list(range(10))
    test_indices = list(range(5))

    # 훈련 및 테스트 인덱스를 파일로 저장
    with open(log_path + 'trainindex.txt', 'w') as f:
        f.write("\n".join([str(x) for x in train_indices]))

    with open(log_path + 'testindex.txt', 'w') as f:
        f.write("\n".join([str(x) for x in test_indices]))

    # 정렬된 이미지 및 라벨 경로 리스트 생성
    train_img_path = sorted(glob.glob(train_img_dir + '/*.nii.gz'))
    train_gt_path = sorted(glob.glob(train_gt_dir + '/*.nrrd'))
    test_img_path = sorted(glob.glob(test_img_dir + '/*.nii.gz'))
    test_gt_path = sorted(glob.glob(test_gt_dir + '/*.nrrd'))

    # 훈련 및 테스트 이미지와 라벨 경로 리스트에서 해당 인덱스만 선택
    train_img_path = [train_img_path[i] for i in train_indices]
    train_gt_path = [train_gt_path[i] for i in train_indices]
    test_img_path = [test_img_path[i] for i in test_indices]
    test_gt_path = [test_gt_path[i] for i in test_indices]

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
        # tio.Resample((512, 512, 36)),

        tio.CropOrPad(size_base),  # 기본 크기로 이미지를 자르거나 채웁니다.
        # tio.RandomFlip(),  # 랜덤으로 이미지를 뒤집습니다.
        tio.RandomAffine(default_pad_value=0),  # 랜덤한 어파인 변환을 적용합니다.
        tio.RandomBiasField(),  # 랜덤한 바이어스 필드를 추가합니다.

        # tio.Resize(size_resize),  # 이미지의 크기를 재조정합니다.
        # tio.CropOrPad(size_crop),  # 이미지를 잘라내거나 크기를 조정합니다.

        # tio.RandomMotion(),  # 랜덤 모션을 추가합니다.
        # tio.RandomAnisotropy(),  # 랜덤한 이방성을 추가합니다.
        # tio.RandomElasticDeformation(),  # 랜덤한 탄성 변형을 추가합니다.
        # tio.RandomGhosting(),  # 랜덤한 유령 효과를 추가합니다.
        # tio.RandomSpike(),  # 랜덤한 스파이크를 추가합니다.
        tio.RandomNoise(),  # 랜덤한 노이즈를 추가합니다.
        # tio.RandomBlur(),  # 랜덤한 블러 효과를 추가합니다.
        # tio.RandomSwap(),  # 랜덤하게 픽셀을 교환합니다.
        tio.RandomGamma(),  # 랜덤한 감마 변환을 적용합니다.
        # tio.Pad((patch_size//2) + 1),  # 패딩을 추가합니다.
    ])

    # 이전 코드와 동일하지만, 랜덤한 변환들을 제외하고, 이미지를 재조정하고 잘라내는 단계만 포함합니다.
    test_transform = tio.Compose([
        # 2023.05.23
        # - train loader의 이미지가 같은 공간에 있지 않다는 에러에 대한 대응
        # tio.ToCanonical(),
        # tio.Resample((512, 512, 36)),

        tio.CropOrPad(size_base),
        tio.Resize(size_resize),
        tio.CropOrPad(size_crop),
    ])

    # Set dataset
    train_set = tio.SubjectsDataset(
        train_subjects, transform=train_transform)  # 훈련 데이터셋 설정

    test_set = tio.SubjectsDataset(
        test_subjects, transform=test_transform)  # 테스트 데이터셋 설정

    visualize = False
    if visualize:
        import matplotlib.pyplot as plt

        # Get the first subject from the dataset
        subject = train_set[8]

        # Get the image and label tensors
        image_tensor = subject['img']['data']
        label_tensor = subject['gt']['data']

        image_tensor = image_tensor.squeeze()
        label_tensor = label_tensor.squeeze()

        # Convert the tensors to numpy arrays
        image_array = image_tensor.numpy()
        label_array = label_tensor.numpy()



        # Plot the middle slice of the image and label
        middle_slice = image_array.shape[2] // 2

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(image_array[:, :, middle_slice], cmap='gray')
        plt.title('Image')

        plt.subplot(1, 2, 2)
        plt.imshow(label_array[:, :, middle_slice], cmap='gray')
        plt.title('Label')

        plt.show()

    # 여기서는 균일한 샘플링을 사용합니다. 주석 처리된 코드는 레이블 별 확률을 지정하여 샘플링하는 코드입니다.
    train_sampler = tio.data.UniformSampler(patch_size)
    # train_sampler = tio.data.LabelSampler(patch_size, label_probabilities={0: 0.5, 1: 0.5})

    # 이 Queue 객체는 훈련 중에 훈련 패치를 효율적으로 로드하는데 사용됩니다.
    patches_train_set = tio.Queue(
        subjects_dataset=train_set,             # 환자 기준 데이터
        max_length=max_queue_length,            # patch 큐의 최대 길이 (?)
        samples_per_volume=samples_per_volume,  # 각 환자에서 추출할 패치의 수
        sampler=train_sampler,                  # 패치 사이즈가 결정되어 있는, 샘플러
        num_workers=16,
        shuffle_subjects=False,
        shuffle_patches=True,
    )

    if visualize:
        for i in range(100):
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

            # Plot the middle slice of the image and label
            middle_slice = image_array.shape[2] // 2

            if label_array[:, :, middle_slice].max() > 0:
                print(label_array[:, :, middle_slice].max())

                plt.figure(figsize=(12, 6))

                plt.subplot(1, 2, 1)
                plt.imshow(image_array[:, :, middle_slice], cmap='gray')
                plt.title('Image')

                plt.subplot(1, 2, 2)
                plt.imshow(label_array[:, :, middle_slice], cmap='gray')
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
    criterion_b = MyDiceLoss()


    loss_function = TverskyLoss(to_onehot_y=True, softmax=False, alpha=alpha, beta=beta, smooth_dr=1e-6, smooth_nr=1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # 학습률 스케줄러 설정
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.95 ** epoch)

    # 스케일러는 부동소수점 연산의 오버플로우와 언더플로우를 관리합니다.
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # 로그를 작성하기 위한 writer를 생성합니다.
    writer = SummaryWriter()

    # 현재 시간을 기록하여 훈련 시간을 측정합니다.
    start_time = time.time()

    # 초기 최적 손실 값을 설정합니다.
    epoch_loss_best = [0, 1.0]  # [epoch, value_best]
    epoch_dice_best = [0, 1.0]  # [epoch, value_best]

    # 여기서는 훈련과 검증을 실행하는 메인 루프가 시작됩니다.
    for epoch in range(epochs):

        time_startepoch = time.time()

        model.train()
        epoch_loss = 0
        epoch_dice = 0
        epoch_numbest = 0

        # 배치마다 데이터를 로드하고, 모델을 통해 예측을 수행하고, 손실을 계산하고, 가중치를 업데이트합니다.
        for i, data in enumerate(train_loader_patches):
            with torch.cuda.amp.autocast():
                x, target = data['img']['data'].to(device), data['gt']['data'].to(device)
                output = model(x)
                output = torch.softmax(output, axis=1)
                target = 1 * (target > 0)
                loss = loss_function(output, target)
                output_thr = 1. * (output > 0.5)
                dice = criterion_b(y_true=target, y_pred=output_thr)

            reduce_loss = loss.data.cpu().numpy()
            reduce_dice = dice.data.cpu().numpy()
            epoch_loss += reduce_loss
            epoch_dice += reduce_dice

            logging.info(
                'Epoch:{}_Iter:{} tversky: {:.8f} dice: {:.8f} train'
                    .format(epoch, i, reduce_loss, reduce_dice))

            # 가중치 업데이트
            optimizer.zero_grad()
            scaler.scale(loss + dice).backward()
            scaler.step(optimizer)
            scaler.update()

        # 스케줄러 단계 수행
        scheduler.step()
        time_endepoch = time.time()

        num_train = len(train_set)
        logging.info(
            'Epoch:{}_NumTrainBatch:{} tversky: {:.8f} dice: {:.8f}'
                .format(epoch, num_train, epoch_loss / i, epoch_dice / i))

        # loss와 dice 값을 writer에 추가합니다.
        writer.add_scalar('lr:', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('loss:', reduce_loss, epoch)
        writer.add_scalar('dice:', reduce_dice, epoch)

        # 모델 검증 단계
        model.eval()
        with torch.no_grad():
            epoch_loss = 0
            epoch_dice = 0
            select_loss = 0
            select_dice = 0
            select_list = [0, 3, 6, 7, 8, 11]

            # 모든 테스트 셋을 순회하면서 GridSampler를 사용해 테스트 패치를 생성하고, 패치를 모델에 통과시키며, 결과를 GridAggregator에 추가합니다.
            for i, data in enumerate(test_set):
                # GridSampler를 생성하여 이미지를 겹치는 패치로 나눕니다.
                grid_sampler = tio.inference.GridSampler(data, patch_size,
                                                         patch_overlap=(patch_size * overlap_numer) // overlap_denom)

                # DataLoader를 생성하여 배치 형태로 데이터를 로드합니다.
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=2)

                # GridAggregator를 생성하여 각 패치에 대한 출력을 결합합니다.
                aggregator = tio.inference.GridAggregator(grid_sampler)

                # 각 배치에 대해 모델을 실행하고 결과를 aggregator에 추가합니다.
                for patches_batch in patch_loader:
                    input_tensor = patches_batch['img'][tio.DATA].to(device)
                    locations = patches_batch[tio.LOCATION]
                    outputs = model.forward(input_tensor)
                    aggregator.add_batch(outputs, locations)

                # GridAggregator에서 출력 텐서를 얻습니다.
                output_tensor = aggregator.get_output_tensor()

                # 소프트맥스를 적용하고 차원을 추가합니다.
                output_tensor = torch.unsqueeze(torch.softmax(output_tensor, axis=0), dim=0)

                # 대상 텐서에 차원을 추가합니다.
                target = torch.unsqueeze(data['gt']['data'], dim=0)

                # 출력 텐서에 임계값을 적용하고, 대상 텐서에 이진화를 적용합니다.
                output_tensor_thr = 1. * (output_tensor > 0.5)
                target = 1 * (target > 0)

                # 손실을 계산합니다.
                loss = loss_function(output_tensor_thr, target)

                # 다이스 계수를 계산합니다.
                dice = criterion_b(y_true=target, y_pred=output_tensor_thr)

                # 손실과 다이스 계수를 CPU로 이동하고 numpy 배열로 변환합니다.
                reduce_loss = loss.data.cpu().numpy()
                reduce_dice = dice.data.cpu().numpy()

                # 에폭에 대한 총 손실과 다이스 계수를 누적합니다.
                epoch_loss += reduce_loss
                epoch_dice += reduce_dice

                # 선택된 배치에 대한 손실과 다이스 계수를 누적합니다.
                if i in select_list:
                    select_loss += reduce_loss
                    select_dice += reduce_dice

                # 로깅 메시지를 생성합니다.
                logging.info(
                    'Epoch:{}_Iter:{} tversky: {:.8f} dice: {:.8f} test fname:{} seed:{}'
                        .format(epoch, i, reduce_loss, reduce_dice, test_gt_path[i], ranseed))

                # 결과 이미지를 저장하는 과정입니다.
                sname = test_gt_path[i].split('/')[-1].split('.')[0]

                orig_dir = test_gt_dir
                orig_fpath = os.path.join(orig_dir, sname + '.nii.gz')
                orig_nii = nib.load(orig_fpath)
                orig_img = orig_nii.get_fdata()
                orig_size = orig_img.shape
                transform_test = tio.Compose([
                    tio.CropOrPad(size_resize),
                    tio.Resize(size_base),
                    tio.CropOrPad(orig_size),
                ])
                subj = tio.Subject(gt=tio.LabelMap(tensor=output_tensor[0, 1:2, :, :, :].cpu().numpy()))
                subj = transform_test(subj)

                recon_img = subj['gt']['data'][0].cpu().numpy()
                recon_nii = nib.Nifti1Image(recon_img, orig_nii.affine, orig_nii.header)

                recon_fpath = os.path.join(recon_dir, sname + '_pred.nii.gz')
                nib.save(recon_nii, recon_fpath)

            # 테스트 세트의 크기를 계산합니다.
            num_test = len(test_set)
            # 에폭에 대한 평균 손실과 다이스 계수를 계산합니다.
            epoch_loss_current = epoch_loss / num_test
            epoch_dice_current = epoch_dice / num_test

            # 선택된 배치에 대한 평균 손실과 다이스 계수를 계산합니다.
            select_loss_current = select_loss / num_test
            select_dice_current = select_dice / num_test

            # 최고의 다이스 계수를 확인하고 모델을 저장합니다.
            if select_dice_current < epoch_dice_best[1]:
                epoch_loss_best = [epoch, select_loss_current]
                epoch_dice_best = [epoch, select_dice_current]

                file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                }, file_name)

                # 이전 체크포인트를 삭제합니다.
                ckpt_rm_list = [x for x in sorted(glob.glob(os.path.join(checkpoint_dir, '*.pth'))) if
                                '_{}.pth'.format(epoch) not in x]
                for ckpt_rm in ckpt_rm_list:
                    os.remove(ckpt_rm)

            # 로깅 메시지를 생성합니다.
            logging.info(
                'Epoch:{}_NumtestBatch:{} tversky: {:.8f} dice: {:.8f} - best: {},{:.8f}'
                    .format(epoch, num_test,
                            epoch_loss_current, epoch_dice_current,
                            epoch_dice_best[0], epoch_dice_best[1]))

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