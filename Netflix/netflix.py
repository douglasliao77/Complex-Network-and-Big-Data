import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from matplotlib import pyplot as plt
from load_data import load_data, getA

nr_users = 2000
nr_movies = 1500
upper_bound = 5
lower_bound = 1

def train_baseline(training_data):
    """
    Uses the provided dataset to train the baseline predictor. 
    Should return three things:
    r_bar: the average rating over all users and movies
    bu: vector where the i:th entry represents the bias of the i:th user compared to r_bar
    bm: vector where the i:th entry represents the bias of the i:th movie compared to r_bar
    """

    A = getA(training_data)
    r_bar = training_data[:, 2].sum() / training_data.shape[0]
    c = training_data[:, 2] - r_bar
    b = scipy.sparse.linalg.lsqr(A, c)[0]
    bu = b[:nr_users] # Bias of user u compared to average r_bar
    bm = b[nr_users:] # Bias of movie m compared to the average r_bar

    return r_bar,bu,bm

def baseline_prediction(training_data,datasets_to_predict):
    """
    Uses the training_data to train the baseline predictor, 
    then evaluates its performance on all the datasets in the test_datas list 
    """
    r_bar,bu,bm = train_baseline(training_data)
    
    # Create a list with one element for each dataset in datasets_to_predict.
    # Each entry (r_hat) should be an array with the predicted ratings for all pairs of users and movies in that dataset
    r_hats = []
    for data in datasets_to_predict:
        users = data[:,0] 
        movies = data[:,1]
        r_hat = np.clip(r_bar + bu[users] + bm[movies], lower_bound, upper_bound)
        r_hats.append(r_hat)

    return r_hats

def neighborhood_prediction(training_data,datasets_to_predict,u_min=1,L=nr_movies):
    """
    Uses the training_data to train the improved predictor, 
    then evaluates its performance on all the datasets in the test_datas list 
    """

    # ========================== Create the cosine similarity matrix D ==========================
    r_bar,bu,bm = train_baseline(training_data)
    users = training_data[:,0] 
    movies = training_data[:,1]
    rating = training_data[:,2]
    r_tilde = np.zeros((nr_users, nr_movies))
    R = np.zeros((nr_users, nr_movies))
    for u,m,r in training_data:
        r_tilde[u][m] = r - np.clip(r_bar+bu[u]+bm[m],lower_bound,upper_bound)
        R[u][m] = True

    
    D = np.zeros((nr_movies, nr_movies))
    for i in range(nr_movies):
        for j in range(i,nr_movies):
            u = np.nonzero(np.logical_and((R[:,i]),(R[:,j])))
            if i == j:
                D[i][j] = 1
            elif u[0].size < u_min:
                D[i][j] = 0
            else:
                num = np.multiply(r_tilde[:,i], r_tilde[:,j])
                den = np.linalg.norm(r_tilde[:,i][u]) * np.linalg.norm(r_tilde[:,j][u])
                D[i][j] = np.sum(num) / den
            D[j][i] = D[i][j] 

    # Uncomment the following lines to check the correctness of the D matrix for u_min = 20 in the verification dataset
    # You should get an error that is less than 1e-5
    # print(np.load('verification_D_mat.npy'))
    if filename == "verification" and u_min == 20:
        error_in_D = np.linalg.norm(np.load('verification_D_mat.npy') - D)
        print("Error in D matrix: {0:.5f}\n".format(error_in_D))

    # =================== Evaluate the performance of the improved predictor ====================
    r_hats = []
    for data in datasets_to_predict:
        r_hat = np.zeros(len(data)) 

        for idx, (u,i,_) in enumerate(data):
            # Sort by similiarity for movie i and remove itself
            L_set = np.argsort(D[i])[::-1][:L]
            # Calculate the extra weight
            num = 0
            den = 0
            for j in L_set:
                if D[i][j] !=0 or R[u][j]!=0:
                    num += D[i][j]*r_tilde[u][j]
                    den += abs(D[i][j])
            weight = num / den if den != 0 else 0
            r_hat[idx] = np.clip((r_bar + bu[u] + bm[i]) + weight, lower_bound, upper_bound)

        r_hats.append(r_hat)
    return r_hats


def RMSE(r_hat,r):
    # Compute the RMSE between the true ratings r and the predicted ratings r_hat
    return np.sqrt(np.mean((r - r_hat) ** 2))

def draw_histogram(r_hat,r,name=""):
    # Create the described histogram
    abs_errors = np.abs(np.round(r_hat) - r)
    plt.figure(1)
    plt.hist(abs_errors,bins=np.arange(0, 5) - 0.5, edgecolor="black")
    plt.title(name)
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.xticks(np.arange(0, 6))
    plt.show()
    return 0


# Load training and test data
filename = "verification"
training_data = load_data(filename+'.training')
test_data = load_data(filename+'.test')


# ====================================== TASK 1 ======================================
print("---- baseline predictor ----")

[r_hat_baseline_training,r_hat_baseline_test] = \
    baseline_prediction(training_data,[training_data,test_data])

rmse_baseline_training = RMSE(r_hat_baseline_training,training_data[:,2])
rmse_baseline_test = RMSE(r_hat_baseline_test,test_data[:,2])

print("Training RMSE: {0:.3f}".format(rmse_baseline_training))
print("Test RMSE: {0:.3f}".format(rmse_baseline_test))

# draw_histogram(r_hat_baseline_test, test_data[:,2],"Baseline Test")


# ====================================== TASK 2 ======================================
u_min = 20
L = 100
print("\n---- movie neighborhood predictor with u_min = {} and L = {} ----".format(u_min,L))

[r_hat_neighborhood_training,r_hat_neighborhood_test] = \
    neighborhood_prediction(training_data,[training_data,test_data],u_min,L)

rmse_neighborhood_training = RMSE(r_hat_neighborhood_training,training_data[:,2])
rmse_neighborhood_test = RMSE(r_hat_neighborhood_test,test_data[:,2])

print("Training RMSE: {0:.3f}".format(rmse_neighborhood_training))
print("Test RMSE: {0:.3f}".format(rmse_neighborhood_test))

print("\nTraining Improvement: {0:.3f}%".format(
    (rmse_baseline_training-rmse_neighborhood_training)/rmse_baseline_training*100))
print("Test Improvement: {0:.3f}%".format(
    (rmse_baseline_test-rmse_neighborhood_test)/rmse_baseline_test*100))


