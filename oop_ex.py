"""
This module provides an example of object oriented programming
"""

import numpy as np

class calendar:

    def __init__(self, yr=1, mth=1, day=1):
        '''
        Initialize a calendar (without leap year) at a given
        year, month, and day.
        '''

        self.year = int(yr)
        self.month = int(mth)
        self.day = int(day)

        # List of days in each month
        self.days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


    # def __repr__(self):
    #     '''
    #     What to show when an object is queried with the repr() function or via
    #     the interactive prompt
    #     '''
    #     return "{}/{}/{}".format(self.month, self.day, self.year)


    def __str__(self):
        '''
        What to show when the print statement is called on this class.
        '''
        return "Day {} of Month {} in Year {}".format(self.day, self.month, self.year)


    def advance(self):
        '''
        Advance one calendar day.
        '''

        self.day += 1
        # check for new month
        if self.day > self.days_in_month[self.month-1]:
            # replace with correct day, update month
            self.day = 1
            self.month += 1
            # now need to also check for new year
            if self.month > 12:
                # replace with correct month, update year
                self.month = 1
                self.year += 1


    def get_day_of_year(self):
        '''
        Return the day of the year
        '''
        cum_days = np.cumsum(self.days_in_month)
        if self.month > 1:
            return self.day + cum_days[self.month-2]
        else:
            return self.day



class cal_plus(calendar):
    '''Builds on the calendar example above, demonstrating inheritance'''

    def __init__(self, yr=1, mth=1, day=1):
        '''Adds a season to the initialization'''

        # first, call the __init__ of the superclass
        super().__init__(yr, mth, day)

        # now build on it
        self.season_list = ['winter', 'winter', 'spring', 'spring', 'spring',\
              'summer', 'summer', 'summer', 'fall', 'fall', 'fall', 'winter']
        

    def advance(self, n=1):
        '''Overrides the base advance class with ability to jump several days
        at a time. n is the number of days to advance
        '''

        n = int(n)
        assert n > 0, "n must be a positive integer"

        self.day += n
        # check for new month in loop
        while self.day > self.days_in_month[self.month-1]:
            # reduce days, update month
            self.day -= self.days_in_month[self.month-1]
            self.month += 1
            # check for new year
            if self.month > 12:
                # replace with correct month, update year
                self.month = 1
                self.year += 1


    def get_season(self):
        '''Return the current season'''

        return self.season_list[self.month-1]
