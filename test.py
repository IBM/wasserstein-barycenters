import numpy as np
import torch
import barycenter
import sys
import argparse

# ------------------------------
# Logger
# ------------------------------           
import logging
import logging.handlers
# logger
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
# formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s(%(name)s): %(message)s')
# console handler
consH=logging.StreamHandler(sys.stdout)
consH.setFormatter(formatter)
consH.setLevel(logging.INFO)
# add the handlers to the logger
logger.addHandler(consH) 
request_file_handler=False
if request_file_handler:
    # file handler
    base=os.path.basename(__file__)
    fileH=logging.handlers.RotatingFileHandler('{}.log'.format(base), maxBytes=10*1024*1024, backupCount=1)
    fileH.setFormatter(formatter)
    fileH.setLevel(logging.INFO)
    # add handler
    logger.addHandler(fileH)
log = logger 

# versions
log.info('python : {}'.format(sys.version))
log.info('torch : {}'.format(torch.__version__))
log.info('numpy : {}'.format(np.version.version))


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--lambda_unbalanced', type=int, help=' -1: balanced, >0: unbalanced',              default=-1)
    parser.add_argument('--stabilization',     type=str, help='original, log-domain, log-stabilized',       default='original')
    parser.add_argument('--tau_acceleration',  type=int, help='0: no acceleration, >0: accelerated',        default=0)
    parser.add_argument('--epsilon',           type=int, help='entropic regularization',                    default=1e-3)
    parser.add_argument('--min_epsilon',       type=int, help='minimum entropic regularization',            default=1e-5)
    parser.add_argument('--reduce_epsilon',    type=int, help='decrease epsilon across iterations',         default=0)
    parser.add_argument('--num_iterations',    type=int, help='maximum number of iterations',               default=100)
    parser.add_argument('--stop_threshold',    type=int, help='stopping threshold on barycenter variation', default=1e-6)
    parser.add_argument('--stab_threshold',    type=int, help='threshold for absorbing U and V',            default=100)
    parser.add_argument('--logging',           type=int, help='save logging info',                          default=0)

    args = parser.parse_args()

    return args



def main():

    args = parse_arguments()

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    #
    # Generate data
    # -------------

    # %% parameters

    n = 100  # nb bins

    # bin positions
    x = np.arange(n, dtype=np.float64)

    # Gaussian distributions
    means = np.array([20., 30., 70.])
    stds  = np.array([2., 5., 2.])
    gauss_hist = []
    
    for mu, sigma in zip(means,stds):
        gauss_samples = sigma * np.random.randn(100) + mu

        log.info("# gauss {} {} : {}".format(mu, sigma, gauss_samples.shape))
        gauss_hist.append(gauss_samples)

        
    # creating matrix A containing all distributions
    A = np.vstack(gauss_hist).T
    log.info("# A : {}".format(A.shape))

    
    def euclidean_distances(X, Y, squared=False):
        XX = np.einsum('ij,ij->i', X, X)[:, np.newaxis]
        YY = np.einsum('ij,ij->i', Y, Y)[np.newaxis, :]
        distances = np.dot(X, Y.T)
        distances *= -2
        distances += XX
        distances += YY
        np.maximum(distances, 0, out=distances)
        if X is Y:
            # Ensure that distances between vectors and themselves are set to 0.0.
            # This may not be the case due to floating point rounding errors.
            distances.flat[::distances.shape[0] + 1] = 0.0
        return distances if squared else np.sqrt(distances, out=distances)

    from scipy.spatial.distance import cdist
    def dist(x1, x2=None, metric='sqeuclidean'):
        if x2 is None:
            x2 = x1
        if metric == "sqeuclidean":
            return euclidean_distances(x1, x2, squared=True)
        return cdist(x1, x2, metric=metric)

    
    def dist0(n):
        x = np.arange(n, dtype=np.float64).reshape((n, 1))
        res = dist(x, x)
        return res
    
    # loss matrix + normalization
    M = dist0(n)
    M /= M.max()
    log.info("# M : {}".format(M.shape))

    #
    # Barycenter computation
    # ----------------------

    # normalized weights
    weights = np.array([1, 1, 1])
    weights = weights / np.sum(weights)

    P = torch.from_numpy(A).double().to(device)
    C = torch.from_numpy(M).double().to(device)
    w = torch.from_numpy(weights).double().to(device).view(1, -1)

    log.info("# P : {}".format(P.size()))
    log.info("# C : {}".format(C.size()))
    log.info("# w : {}".format(w.size()))

    q, logs = barycenter.compute(P,
                                 C,
                                 weights=w,
                                 epsilon=args.epsilon,
                                 stabilization=args.stabilization,
                                 min_epsilon=args.min_epsilon,
                                 reduce_epsilon=args.reduce_epsilon,
                                 lambda_unbalanced=args.lambda_unbalanced,
                                 stab_threshold=args.stab_threshold,
                                 tau_acceleration=args.tau_acceleration,
                                 num_iterations=args.num_iterations,
                                 stop_threshold=args.stop_threshold,
                                 logging=args.logging)
    
    q = q.cpu().numpy()

    log.info('q = {} '.format(q))
    log.info('sum of q = {}'.format(np.sum(q)))

    
if __name__ == "__main__":
    main()
