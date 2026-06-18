from .facmac_learner import FACMACLearner
from .facmac_learner_discrete import FACMACDiscreteLearner
from .EA_facmac_learner import EA_FACMACLearner
from .EA_facmac_learner_discrete import EA_FACMACDiscreteLearner
REGISTRY = {}
REGISTRY["facmac_learner"] = FACMACLearner
REGISTRY["facmac_learner_discrete"] = FACMACDiscreteLearner
REGISTRY["facmac_learner_EA"] = EA_FACMACLearner
REGISTRY["facmac_learner_discrete_EA"] = EA_FACMACDiscreteLearner
