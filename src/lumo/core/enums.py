import enum


class TrainStage(enum.Enum):
    """An enumeration class representing the different stages of training.
    """
    default = 'default'
    train = 'train'
    test = 'test'
    val = 'val'

    def is_train(self):
        """Check if the current stage is the training stage.

        Returns:
            bool: True if the current stage is the training stage, False otherwise.
        """
        return self.value == 'train'

    def is_test(self):
        """Check if the current stage is the testing stage.

        Returns:
            bool: True if the current stage is the testing stage, False otherwise.
        """
        return self.value == 'test'

    def is_val(self):
        """Check if the current stage is the validation stage.

        Returns:
            bool: True if the current stage is the validation stage, False otherwise.
        """
        return self.value == 'val'

    @staticmethod
    def create_from_str(value):
        """Create a TrainStage instance from a string.

        If the value is 'eval' or 'evaluate', it will be converted to 'val'.

        Args:
            value (str): A string representing the stage of training.

        Returns:
            TrainStage: A TrainStage instance representing the stage of training.
        """
        if value in {'eval', 'evaluate'}:
            value = 'val'
        return TrainStage(value)
