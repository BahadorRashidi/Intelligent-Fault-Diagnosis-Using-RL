# %%
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from sklearn.model_selection import StratifiedKFold


class Crosspreprocessing:

    def __init__(self, data_path):
        self.data_path = data_path

    def Data_Preprocess(self, slice_size=1000, L=120, window=False, random=42, divide=0.33):
        '''
            Param=> Data: Data could be DataSetA.mat , DataSetB.mat, DataSetC.mat, DataSetD.mat
            Param=> slice_size: The length of the window
            Param=> L: The stride steping size
            Param=> window: is False: we will have slicing mode with cronological slide_size -
                            is True: We will have strides with window size of "slice_size" and stride length of "L"
        '''
        Data = self.data_path
        mat = spio.loadmat(Data, squeeze_me=True)  # Load the dataset
        data = {}
        Raw = []
        ##-----------import each data set into the data
        for i in range(1, len(mat) - 2):
            data[i - 1] = mat.get('C{}'.format(i))
            # print("Class : ",(i-1)%10, "Imported :)")

        ##-----------When there is no stride
        X_ALL = []
        Y_ALL = []
        if window == False:
            for element in data:
                size = int(len(data[element]) // slice_size)  ## the number of windows in each data
                X = []
                Y = []
                for k in range(size):
                    X.append(np.array(data[element][(k) * slice_size:(k + 1) * slice_size]))
                    Y.append(element % 10)
                X_ALL.append(X)
                Y_ALL.append(Y)
                X = []
                Y = []
        ##-----------When the strides is active
        else:
            for element in data:
                size = int((len(data[element]) - slice_size) / L) + 1  ## number of batches
                X = []
                Y = []
                for k in range(size):
                    X.append(np.array(data[element][int(k * (L)):int(
                        k * L + slice_size)]))  ## for each window it strides with windowsize/2
                    Y.append(element % 10)
                X_ALL.append(X)
                Y_ALL.append(Y)
                print("Class {} has shape : ".format(element % 10), np.shape(X))
                X = []
                Y = []

        Flat_X = []
        Flat_Y = []
        FFT_X = []
        for element in X_ALL:
            for things in element:
                Flat_X.append(things)
                fft_signal = np.fft.fft(things)
                FFT_X.append(abs(fft_signal.real[0:len(fft_signal.real) // 2]))
        for element in Y_ALL:
            for things in element:
                Flat_Y.append(things)
        return np.asarray(normalize(FFT_X)), np.asarray(Flat_Y)
        # kf = StratifiedKFold(n_splits=10, random_state=random, shuffle=True).split(np.asarray(normalize(FFT_X)), np.asarray(Flat_Y))
        # # for train_idx, test_idx in kf:
        # #
        # #
        # #
        # #
        # # X_train, X_test, y_train, y_test = train_test_split(np.asarray(normalize(FFT_X)), np.asarray(Flat_Y),
        # #                                                     test_size=divide, random_state=random, shuffle=True)
        # # print("X_TRAIN has shape:", np.shape(X_train))
        # # print("Y_TRAIN has shape:", np.shape(y_train))
        # # print("X_TEST has shape:", np.shape(X_test))
        # # print("Y_TEST has shape:", np.shape(y_test))
        # # print("Classes are :", np.unique(y_train))
        # # return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    A_preprocessing = Crosspreprocessing('DataSetB.mat')
    # Play with L to increase and decrese the number of rows in your training and test matrices
    X_train, Y_train = A_preprocessing.Data_Preprocess(slice_size=2400, L=2400, window=False)
    # print(X_train.shape, X_test.shape)
    # print(y_train)






