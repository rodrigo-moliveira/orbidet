"""Common errors declarations
"""


class OrbidetError(Exception):
    """Generic error"""

    pass

class MissingDbValue(OrbidetError):
    """Missing value in DataBase
    """
    pass

class ConfigError(OrbidetError):
    """Missing value in DataBase
    """
    pass

class GravityError(OrbidetError):
    """Missing value in DataBase
    """
    pass

class LSnotConverged(OrbidetError):
    """Least Squares did not converge
    """
    pass
