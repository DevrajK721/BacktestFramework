# Common Interface every strategy must follow 

from abc import ABC, abstractmethod # Used for defining abstract base classes (user provides own strategies)
import pandas as pd 

class Strategy(ABC):
    @abstractmethod 
    def generate_positions(self, data: pd.DataFrame) -> pd.Series:
        """Return target positions derived from historical market data.

        Parameters
        ----------
        data:
            Clean daily OHLCV data indexed by date.

        Returns
        -------
        pd.Series
            A Series named ``"target_position"`` with the same index as
            ``data``. Values represent target exposure in the range
            [-1.0, 1.0]: long, flat, or short.

        Notes
        -----
        A strategy must use only information available up to each date.
        It must not shift positions, calculate returns, model costs, or
        calculate performance metrics. The engine owns execution timing.
        """
        target_position = pd.Series() # Initialize series 
        target_position.index = data.index # Align dates 
        target_position. 


