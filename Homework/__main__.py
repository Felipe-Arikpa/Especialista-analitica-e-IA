import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
                      
from .src.predictor import main

if __name__ == '__main__':

    main()