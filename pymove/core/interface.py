import abc


class MoveDataFrameAbstractModel(abc.ABC):
    @abc.abstractmethod
    def lat(self):
        pass

    @abc.abstractmethod
    def lng(self):
        pass

    @abc.abstractmethod
    def datetime(self):
        pass

    @abc.abstractmethod
    def loc(self):
        pass

    @abc.abstractmethod
    def iloc(self):
        pass

    @abc.abstractmethod
    def at(self):
        pass

    @abc.abstractmethod
    def values(self):
        pass

    @abc.abstractmethod
    def columns(self):
        pass

    @abc.abstractmethod
    def index(self):
        pass

    @abc.abstractmethod
    def dtypes(self):
        pass

    @abc.abstractmethod
    def shape(self):
        pass

    @abc.abstractmethod
    def rename(self):
        pass

    @abc.abstractmethod
    def len(self):
        pass

    @abc.abstractmethod
    def head(self):
        pass

    @abc.abstractmethod
    def tail(self):
        pass

    @abc.abstractmethod
    def get_users_number(self):
        pass

    @abc.abstractmethod
    def to_numpy(self):
        pass

    @abc.abstractmethod
    def to_dict(self):
        pass

    @abc.abstractmethod
    def to_grid(self):
        pass

    @abc.abstractmethod
    def to_data_frame(self):
        pass

    @abc.abstractmethod
    def info(self):
        pass

    @abc.abstractmethod
    def describe(self):
        pass

    @abc.abstractmethod
    def memory_usage(self):
        pass

    @abc.abstractmethod
    def copy(self):
        pass

    @abc.abstractmethod
    def generate_tid_based_on_id_datetime(self):
        pass

    @abc.abstractmethod
    def generate_date_features(self):
        pass

    @abc.abstractmethod
    def generate_hour_features(self):
        pass

    @abc.abstractmethod
    def generate_day_of_the_week_features(self):
        pass

    @abc.abstractmethod
    def generate_weekend_features(self):
        pass

    @abc.abstractmethod
    def generate_time_of_day_features(self):
        pass

    @abc.abstractmethod
    def generate_datetime_in_format_cyclical(self):
        pass

    @abc.abstractmethod
    def generate_dist_time_speed_features(self):
        pass

    @abc.abstractmethod
    def generate_dist_features(self):
        pass

    @abc.abstractmethod
    def generate_time_features(self):
        pass

    @abc.abstractmethod
    def generate_speed_features(self):
        pass

    @abc.abstractmethod
    def generate_move_and_stop_by_radius(self):
        pass

    @abc.abstractmethod
    def time_interval(self):
        pass

    @abc.abstractmethod
    def get_bbox(self):
        pass

    @abc.abstractmethod
    def plot_all_features(self):
        pass

    @abc.abstractmethod
    def plot_trajs(self):
        pass

    @abc.abstractmethod
    def plot_traj_id(self):
        pass

    @abc.abstractmethod
    def show_trajectories_info(self):
        pass

    @abc.abstractmethod
    def min(self):
        pass

    @abc.abstractmethod
    def max(self):
        pass

    @abc.abstractmethod
    def count(self):
        pass

    @abc.abstractmethod
    def groupby(self):
        pass

    @abc.abstractmethod
    def plot(self):
        pass

    @abc.abstractmethod
    def select_dtypes(self):
        pass

    @abc.abstractmethod
    def astype(self):
        pass

    @abc.abstractmethod
    def sort_values(self):
        pass

    @abc.abstractmethod
    def reset_index(self):
        pass

    @abc.abstractmethod
    def set_index(self):
        pass

    @abc.abstractmethod
    def drop(self):
        pass

    @abc.abstractmethod
    def duplicated(self):
        pass

    @abc.abstractmethod
    def drop_duplicates(self):
        pass

    @abc.abstractmethod
    def shift(self):
        pass

    @abc.abstractmethod
    def all(self):
        pass

    @abc.abstractmethod
    def any(self):
        pass

    @abc.abstractmethod
    def isna(self):
        pass

    @abc.abstractmethod
    def fillna(self):
        pass

    @abc.abstractmethod
    def dropna(self):
        pass

    @abc.abstractmethod
    def sample(self):
        pass

    @abc.abstractmethod
    def isin(self):
        pass

    @abc.abstractmethod
    def append(self):
        pass

    @abc.abstractmethod
    def join(self):
        pass

    @abc.abstractmethod
    def merge(self):
        pass

    @abc.abstractmethod
    def nunique(self):
        pass

    @abc.abstractmethod
    def to_csv(self):
        pass

    @abc.abstractmethod
    def write_file(self):
        pass

    @abc.abstractmethod
    def convert_to(self):
        pass

    @abc.abstractmethod
    def get_type(self):
        pass
