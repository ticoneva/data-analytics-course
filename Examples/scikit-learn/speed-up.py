# This example demonstrates various ways you can speed up training
# of Scikit-learn models

# Command line arguments
import argparse
parser = argparse.ArgumentParser(description='Various ways to speed up training.')
parser.add_argument('mode', nargs='?', default='default',
                    help='Choose from: default sklearnex gridsearch gridsearch-4 gridsearch-16 gridsearch-sklearnex gridsearch-4-sklearnex gridsearch-16-sklearnex')
parser.add_argument('-N', dest='N', type=int, default=20000,
                    help='Number of samples to generate. Default is 20000.')
args = parser.parse_args()

# Import some libraries
import time
import multiprocessing
import os
import numpy as np

# No. of CPU cores
try: 
    cpus = os.environ['SLURM_CPUS_ON_NODE']
except:
    cpus = str(multiprocessing.cpu_count()) + " (could be fewer)"
print("Available CPUs:",cpus)

# Generate data
print("Generating",args.N,"samples...")
var_num = 10
X = np.random.rand(args.N,var_num).astype('f')
y = np.where(np.sum(X,axis=1)>var_num*0.5,1,0).astype('f')

start = time.time()

if args.mode=='sklearnex':
    # Intel version of SVC
    print("Mode: Intel Extension for Scikit-learn")

    from sklearnex import patch_sklearn
    patch_sklearn()
    from sklearn.svm import SVC
    svc = SVC()
    svc.fit(X,y)
    
elif args.mode=='cuml':
    # CuML not working
    print("Mode: CuML")
    
    from cuml.svm import SVC
    svc = SVC()
    svc.fit(X,y)
    
elif args.mode=='gridsearch-sklearnex':
    # GridSearchCV
    print("Mode: Default GridSearchCV and Intel Extension for Scikit-learn")
    
    from sklearnex import patch_sklearn
    patch_sklearn()
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    parameters = {'C':np.logspace(0.001,10,16)}
    svc = SVC()
    gscv = GridSearchCV(svc,parameters,cv=5)
    gscv.fit(X, y)
    
elif args.mode=='gridsearch-4-sklearnex':
    # GridSearchCV n_jobs=16
    print("Mode: GridSearchCV with n_jobs=4 and Intel Extension for Scikit-learn")   
    
    from sklearnex import patch_sklearn
    patch_sklearn()    
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    parameters = {'C':np.logspace(0.001,10,16)}
    svc = SVC()
    gscv = GridSearchCV(svc,parameters,n_jobs=4,cv=5)
    gscv.fit(X, y)     

elif args.mode=='gridsearch-16-sklearnex':
    # GridSearchCV n_jobs=16
    print("Mode: GridSearchCV with n_jobs=16 and Intel Extension for Scikit-learn")
    
    from sklearnex import patch_sklearn
    patch_sklearn()    
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    parameters = {'C':np.logspace(0.001,10,16)}
    svc = SVC()
    gscv = GridSearchCV(svc,parameters,n_jobs=16,cv=5)
    gscv.fit(X, y)        
    
elif args.mode=='gridsearch':
    # GridSearchCV
    print("Mode: Default GridSearchCV")
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    parameters = {'C':np.logspace(0.001,10,16)}
    svc = SVC()
    gscv = GridSearchCV(svc,parameters,cv=5)
    gscv.fit(X, y)
    
elif args.mode=='gridsearch-4':
    # GridSearchCV n_jobs=16
    print("Mode: GridSearchCV with n_jobs=4")   
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    parameters = {'C':np.logspace(0.001,10,16)}
    svc = SVC()
    gscv = GridSearchCV(svc,parameters,n_jobs=4,cv=5)
    gscv.fit(X, y)     

elif args.mode=='gridsearch-16':
    # GridSearchCV n_jobs=16
    print("Mode: GridSearchCV with n_jobs=16")   
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    parameters = {'C':np.logspace(0.001,10,16)}
    svc = SVC()
    gscv = GridSearchCV(svc,parameters,n_jobs=16,cv=5)
    gscv.fit(X, y)    
    
else:
    # Default SVC
    print("Mode: Scikit-learn")

    from sklearn.svm import SVC
    svc = SVC()
    svc.fit(X,y)

print(str(round(time.time() - start,3))+"s")

   
    
