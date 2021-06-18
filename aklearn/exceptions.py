''' Store all of the various customized exceptions that may be raised.
'''

class InvalidProportion(Exception):
    '''Ensures we use a fraction between 0 and 1.'''
    def __init__(self, message='split_proportion must lie between 0 and 1 (inclusive)'):
        super().__init__(message)

class ConflictingInputSizes(Exception):
    '''Ensures feature matrix and output vector.'''
    def __init__(self, message='split_proportion must lie between 0 and 1 (inclusive)'):
        super().__init__(message)

def ClassInstantiationChecks(features, output, split_proportion, number_labels, standardized):
    if split_proportion < 0 or split_proportion > 1:
            raise InvalidProportion

    if features.shape[0] != output.shape[0]:
        raise ConflictingInputSizes
    