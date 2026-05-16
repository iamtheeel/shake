# -*- coding: utf-8 -*-
"""
@author: caicedo, francolj
"""

import sqlite3

class Storage():
    """
    Creates and handles an SQLite database
    
    Attributes
    ----------
    __fname__ : str
        SQLite database filename
    __table__ : str
        Table title for the specified database
    __conn__ : SQLite connect object 
    
    __cursor__ : SQLite cursor object    
        
    Methods
    -------
    open()
        Opens the SQLite database
    add()
        Inserts fields and values into specific fields 
    read()
        Retrieves specific columns and associated values from database
    close()
        Closes the database connection
        
    Example
    -------        
        import storage as st
        tmp = st.Storage('test.db','test','id integer primary key, number double')
        data = {'id':0, 'number':10.3}
        tmp.add(data)
        tmp.read('0','number')
        tmp.close()
    
    """
    def __init__(self, fname = 'experiment_properties', table = "table", fields = "id integer primary key, number double"):
        
        self.__fname__ = fname
        self.__table__ = table
        self.__conn__ = None
        self.__cursor__ = None

    def check(self, table, fields):
        """
        Checks the existance of a table with a particular set of fields
        
        Parameters:
        -----------
        table : str
            Table title
        
        fields : str
            Fields for the table including data type, separated by a comma
        
        Returns:
        --------
        none     
        
        """      
        # create a database connection
        self.__conn__ = sqlite3.connect(self.__fname__, timeout = 30)

        self.__cursor__ = self.__conn__.cursor()

        
        # create tables
        if self.__conn__ is not None:
            # create table if does not exist
            sql = 'create table if not exists %s(%s);'%(table, fields)
            self.__cursor__.execute(sql)
        else:
            print("Error! cannot create the database connection.")
        

    def add(self,table,data):
        """
        Add data stored in a dictionary
        
        Parameters:
        -----------
        table : str
            Table where the data should be added
        data : dictionary
            Dictionary contaning appropiate fields and values
             
        Returns:
        --------
        none
        """
        fields = ''
        values = ''
        for key in data.keys():
            fields = fields + '"%s", '%key
            values = values + '%s, '%data[key]
        
        fields = fields[0:-2]
        values = values[0:-2]
        sql = 'insert into %s(%s) values (%s);'%(table, fields, values)
        self.__cursor__.execute(sql)
        self.__conn__.commit()
        
    def addmany(self,table,datalist):
        """
        Adds many rows to the database.  The data expected is a list of dictionaries
        where each dictionary describes the data to be added

        Parameters
        ----------
        table : str
            Table where the data should be added
        datalist : list
            List of dictionaries.  Each dictionary will have the name of the
            column (key) and the value to be added (dictionary value)

        Returns
        -------
        None.

        """
        
        fields = ''
        valfields = ''
        for key in datalist[0].keys():
            fields = fields + '"%s", '%key
            valfields = valfields + ':%s, '%key
        fields = fields[0:-2]
        valfields = valfields[0:-2]
        
        
        # See example here: https://stackoverflow.com/questions/18219779/bulk-insert-huge-data-into-sqlite-using-python
        self.__conn__.executemany('insert into %s(%s) values(%s)'%(table,fields,valfields),tuple(datalist))
        self.__conn__.commit()
        

    def read_unique(self,table,col='id'):
        """
        Return a list of unique values on a particular column.  The default
        column is "id".  This can be used to determine if all trials have
        gone through a particular process.

        Parameters
        ----------
        table : str
            Table where the data should be added
        col : str, optional
            Column used to return unique values. The default is 'id'.

        Returns
        -------
        data : list
            List of unique values

        """
        sql = 'select distinct %s from %s order by %s'%(col, table, col)
        res = self.__cursor__.execute(sql)
        
        return  res.fetchall()
    
    def read_all(self,table):
        '''
        Read all the values on a table.

        Parameters
        ----------
        table : str
            Table in the database to read parameters

        Returns
        -------
        data : tupple 
            structure with matching data from database

        '''
        sql = 'select * from %s'%(table)
        res = self.__cursor__.execute(sql)       
        
        return  res.fetchall()

    
    def read(self,table,col,value):
        """
        Reads and retrieve data stored in a database by value
        
        Parameters:
        -----------
        table : str
            Table to read the parameters
            
        col : str
            Column identifier
            
        value : str
            Value from column
            
        Returns:
        --------
        data : tupple structure with matching data from database
        
        """
        sql = 'select * from %s where "%s" = %s'%(table,col, value)
        res = self.__cursor__.execute(sql)
        
        
        return  res.fetchall()
    
    def drop_table(self,table):
        '''
        Drops a table from the database

        Parameters
        ----------
        table : str
            Table to be dropped.

        Returns
        -------
        None.

        '''
        sql = 'drop table %s'%table
        self.__cursor__.execute(sql)
    
    def close(self):
        """
        Close active database connection
        
        Parameters:
        -----------
        none
            
        Returns:
        --------
        none
        
        """
        self.__conn__.close()
        
    def __del__(self):
        """
        Close active database
        
        Parameters:
        -----------
        none
            
        Returns:
        --------
        none
        
        """
        self.close()