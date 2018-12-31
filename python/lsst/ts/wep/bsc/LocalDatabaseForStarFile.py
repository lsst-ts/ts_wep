import os
import numpy as np

from lsst.ts.wep.bsc.LocalDatabase import LocalDatabase


class LocalDatabaseForStarFile(LocalDatabase):

    PRE_TABLE_NAME = "StarTable"

    def createTable(self, filterType):
        """Create the table in database.

        Parameters
        ----------
        filterType : FilterType
            Filter type.

        Raises
        ------
        ValueError
            Table exists in database already.
        """

        tableName = self._getTableName(filterType)
        if (self._tableIsInDb(tableName)):
            raise ValueError("%s exists in database already." % tableName)

        # Create the table
        command = "CREATE TABLE %s" % tableName
        command += "(id INTEGER PRIMARY KEY, simobjid INTEGER NOT NULL, "
        command += "ra REAL, decl REAL, %smag REAL, bright_star NUMERIC)" \
                   % filterType.name.lower()
        self.cursor.execute(command)

        # Commit the change to database
        self.connection.commit()

    def _tableIsInDb(self, tableName):
        """Check the specific table exists in the database or not.

        Parameters
        ----------
        tableName : str
            Table name.

        Returns
        -------
        bool
            Table exists or not.
        """

        # Query the specific table name
        command = "PRAGMA table_info(%s)" % tableName
        self.cursor.execute(command)
        data = self.cursor.fetchall()

        # Check the table exists or not
        if (len(data) == 0):
            return False
        else:
            return True

    def insertDataByFile(self, skyFilePath, filterType, skiprows=1):
        """Insert the sky data by file.

        Parameters
        ----------
        skyFilePath : str
            Sky data file path.
        filterType : FilterType
            Filter type.
        skiprows : int, optional
            Skip the first 'skiprows' lines. (the default is 1.)
        """

        # Get the data
        skyData = np.loadtxt(skyFilePath, skiprows=skiprows)

        # Add the star
        tableName = self._getTableName(filterType)
        for ii in range(len(skyData)):
            # Insert data
            command = "INSERT INTO " + tableName + \
                      " (simobjid, ra, decl, " + \
                      filterType.name.lower() + "mag, bright_star) " + \
                      "VALUES (?, ?, ?, ?, ?)"

            simobjID, ra, decl, mag = skyData[ii]
            task = (int(simobjID), ra, decl, mag, 0)

            self.cursor.execute(command, task)

        # Commit the change to database
        self.connection.commit()

    def deleteTable(self, filterType):
        """Delete the table in database.

        Parameters
        ----------
        filterType : FilterType
            Filter type.
        """

        # Delete the table
        tableName = self._getTableName(filterType)
        command = "DROP TABLE IF EXISTS %s" % tableName
        self.cursor.execute(command)

        # Commit the change to database
        self.connection.commit()


if __name__ == "__main__":
    pass
