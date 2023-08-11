from torch import nn


class MyFocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.3):
        super(MyFocalTverskyLoss, self).__init__()
        #self.sigmoid = nn.Sigmoid()

        self.alpha = alpha
        self.beta = 1 - self.alpha
        self.gamma = gamma

    def forward(self, y_true, y_pred):

        smooth = 1e-6

        y_pred = y_pred[:, 1, :, :, :]
        y_true = y_true[:, 0, :, :, :] # 0 means, the one-hot conversion was not applied

        TP = (y_true * y_pred).sum()
        FP = ((1 - y_true) * y_pred).sum()
        FN = (y_true * (1 - y_pred)).sum()

        numerator = TP
        denominator = TP + (self.beta * FN) + (self.alpha * FP)

        #print('TVERSKY CHECK TP, FN, FP', TP.data.item(), FN.data.item(), FP.data.item())
        #print('TVERSKY CHECK NUMER, DENOM', numerator.data.item(), denominator.data.item())

        tversky = (numerator + smooth) / (denominator + smooth)
        tversky_loss = 1 - tversky
        focal_tversky_loss = tversky_loss ** self.gamma

        return focal_tversky_loss


class MyDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MyDiceLoss, self).__init__()

    def forward(self, y_true, y_pred):

        smooth = 1e-6

        y_pred_Hemo = y_pred[:, 0, :, :, :]
        y_true_Hemo = y_true[:, 0, :, :, :] # 0 means, the one-hot conversion was not applied

        TP = (y_true_Hemo * y_pred_Hemo).sum()
        TN = ((1 - y_true_Hemo) * (1 - y_pred_Hemo)).sum()
        FP = ((1 - y_true_Hemo) * y_pred_Hemo).sum()
        FN = (y_true_Hemo * (1 - y_pred_Hemo)).sum()

        numerator = 2 * TP
        denominator = 2 * TP + FN + FP

        # Dice loss calculation
        dice = (numerator + smooth) / (denominator + smooth)
        dice_loss = 1 - dice

        # Accuracy calculation
        accuracy = (TP + TN) / (TP + TN + FP + FN + smooth)

        # Sensitivity (also known as recall) calculation
        sensitivity = TP / (TP + FN + smooth)

        # Specificity calculation
        specificity = TN / (TN + FP + smooth)

        return dice_loss, accuracy, sensitivity, specificity

# 2023.06.30
# - Batch 별 Dice Loss를 개별적으로 계산하도록 변경
class SegMetrics_sumin(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SegMetrics_sumin, self).__init__()

    def forward(self, y_true, y_pred):

        smooth = 1e-6
        batch_size = y_true.size(0)

        dice_loss = 0
        accuracy = 0
        sensitivity = 0
        specificity = 0

        for i in range(batch_size):
            y_pred_Hemo = y_pred[i, 1, :, :, :]
            y_true_Hemo = y_true[i, 0, :, :, :]

            TP = (y_true_Hemo * y_pred_Hemo).sum()
            TN = ((1 - y_true_Hemo) * (1 - y_pred_Hemo)).sum()
            FP = ((1 - y_true_Hemo) * y_pred_Hemo).sum()
            FN = (y_true_Hemo * (1 - y_pred_Hemo)).sum()

            numerator = 2 * TP
            denominator = 2 * TP + FN + FP

            # Dice loss calculation
            dice = (numerator + smooth) / (denominator + smooth)
            dice_loss += (1 - dice)

            # Accuracy calculation
            accuracy += (TP + TN) / (TP + TN + FP + FN + smooth)

            # Sensitivity (also known as recall) calculation
            sensitivity += TP / (TP + FN + smooth)

            # Specificity calculation
            specificity += TN / (TN + FP + smooth)

        return dice_loss / batch_size, accuracy / batch_size, sensitivity / batch_size, specificity / batch_size