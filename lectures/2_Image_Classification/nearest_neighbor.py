from tqdm import tqdm
import numpy as np

class NearestNeighbor(object):
	def __init__(self):
		pass

	def train(self, X, y):
		# X는 각 행이 example인 N x D 배열
		# y는 크기가 N인 1차원 배열
		# Nearest Neighbor classifier는 단순히 모든 training data를 기억해서 학습한다.
		self.Xtr = X
		self.ytr = y
	
	# L1 distance를 사용해서 predict
	def predict_L1(self, X):
		# X는 우리가 label을 예측하고 싶어하는 각 행이 example인 N x D 배열이다.
		num_test = X.shape[0]	# test case 개수
		
		# output 형태는 train할 때의 input 형태와 동일하게 한다.
		Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

		# 모든 test case에 대해 반복
		for i in tqdm(range(num_test)):
			# L1 distance를 사용해서 i번째 test image와 가장 가까운 train image 찾기
			distances = np.sum(np.abs(self.Xtr - X[i, :]), axis = 1)
			min_index = np.argmin(distances)	# 거리가 가장 가까운 image의 index
			Ypred[i] = self.ytr[min_index]
		
		return Ypred
	
	# L2 distance를 사용해서 predict
	def predict_L2(self, X):
		# X는 우리가 label을 예측하고 싶어하는 각 행이 example인 N x D 배열이다.
		num_test = X.shape[0]	# test case 개수
		
		# output 형태는 train할 때의 input 형태와 동일하게 한다.
		Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

		# 모든 test case에 대해 반복
		for i in tqdm(range(num_test)):
			# L2 distance를 사용해서 i번째 test image와 가장 가까운 train image 찾기
			distances = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis = 1))
			min_index = np.argmin(distances)	# 거리가 가장 가까운 image의 index
			Ypred[i] = self.ytr[min_index]
		
		return Ypred
	
	# K-Nearest Neighbor를 위한 predict 함수
	def predict_knn(self, X, k=1):
		num_test = X.shape[0]

		Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

		# 모든 test case에 대해 반복
		for i in tqdm(range(num_test)):
			# L1 distance를 사용해서 i번째 test image와 가장 가까운 train image 찾기
			distances = np.sum(np.abs(self.Xtr - X[i, :]), axis = 1)
			closest_index = np.argsort(distances)[:k]	# 거리가 가장 가까운 k개 image의 index
			closest_y = self.ytr[closest_index]		# 가장 가까운 k개의 label들
			counts = np.bincount(closest_y)			# k개 중에서 각 label 별 출현 횟수를 가진 배열 저장
			most_common_label = np.argmax(counts)	# 가장 출현 많이한 label 저장
			Ypred[i] = most_common_label
		
		return Ypred