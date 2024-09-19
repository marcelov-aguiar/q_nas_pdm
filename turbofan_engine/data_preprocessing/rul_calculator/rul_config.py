class RULConfig:
    """
    A configuration class that stores constants and settings for calculating 
    Remaining Useful Life (RUL).

    Attributes
    ----------
    feature_unit_number : str
        The column name representing the unit number in the dataset.
    feature_time : str
        The column name representing the time or operational cycle of the unit.
    target : str
        The column name representing the target RUL value.
    default_max_name : str
        The name of the column that stores the maximum time or cycle for each unit.
    total_rul : str
        The column name for the total RUL to be calculated for each unit.
    """
    def __init__(self, feature_unit_number: str, feature_time: str, target: str, 
                 default_max_name: str, total_rul: str):
        self.feature_unit_number = feature_unit_number
        self.feature_time = feature_time
        self.target = target
        self.default_max_name = default_max_name
        self.total_rul = total_rul