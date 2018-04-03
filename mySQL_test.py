#!/usr/bin/python
# -*- coding: utf-8 -*-
import MySQLdb, cPickle
from time import strftime
from keras.models import load_model
import types
import tempfile
import keras.models

class sql_test():
    def __init__(self):
        self.name = "conv1"
        self.description = "model test"
        self.activation = 2
        self.mse = 0.503
        self.mae = 1.012
        self.appliance = "fridge"
        self.params = 1030000
        self.step = 20000
        self.created_time = strftime('%Y-%m-%d_%H_%M')
        self.user  = "shu-wei"
        self.layer_id = 1
        self.layer_type = "conv1"
        self.output_shape = "(None, 24, 60)"
        self.db = MySQLdb.connect(host="140.115.20.30", port=3306, user="NILM", passwd="NILM", db="NILM_Keras", charset="utf8")
        self.cursor = self.db.cursor()
        self.db.ping(True)

    def insert_model(self):
        add_model = "INSERT INTO Model (name, description, activation, mse, mae, appliance, params, step, created_time, user) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
        self.cursor.execute(add_model, (self.name, self.description, self.activation, self.mse, self.mae, self.appliance, self.params, self.step, self.created_time, self.user))
        self.db.commit()

    def search_model(self):
        search_id = "SELECT id FROM Model WHERE name = %s"
        self.cursor.execute(search_id, (self.name,))
        self.model_id = self.cursor.fetchone()

    def insert_layer(self):
        self.search_model()

        add_layer = "INSERT INTO Layer (model_id, layer_id, layer_type, output_shape, params, description, created_time) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        self.cursor.execute(add_layer, (self.model_id, self.layer_id, self.layer_type, self.output_shape, self.params, self.description, self.created_time))
        self.db.commit()

    def insert_model_blob(self, trained_model):
        trained_model = load_model('/home/nilm/NILM_work/models/ukdale_fridge_2018-03-30_18_26.h5')
        self.make_keras_picklable()
        model_blob = cPickle.dumps(trained_model)


        add_blob = "INSERT INTO Model_Blob (model_id, data) VALUES (%s, %s)"
        self.cursor.execute(add_blob, (self.model_id, model_blob))
        self.db.commit()

    def read_model_blob(self):
        self.search_model()

        sql_cmd = "SELECT data FROM Model_Blob WHERE Model_id = %s"
        self.cursor.execute(sql_cmd, (self.model_id,))
        model_blob = self.cursor.fetchone()
        return model_blob

    def load_model(self):
        self.make_keras_picklable()
        sql_model = self.read_model_blob()
        model = cPickle.loads(sql_model[0])
        return model

    def read_model_info(self):
        self.cursor.execute("SELECT * FROM Model")
        results = self.cursor.fetchall()
        for record in results:
            col1 = record[0]
            col2 = record[1]
            print(col1, col2)

    def disconnect(self):
        self.db.close()

    def test(self):
        #file = h5py.File('/home/nilm/NILM_work/models/ukdale_fridge_2018-03-30_18_26.h5', 'r')
        #file.close()
        print('---')


    def run_test(self):
        """self.insert_model()
        self.insert_layer()
        self.insert_model_blob(None)

        self.read_model_info()
        self.load_model()
        self.disconnect()"""
        self.test()

    def make_keras_picklable(self):
        def __getstate__(self):
            model_str = ""
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
                keras.models.save_model(self, fd.name, overwrite=True)
                model_str = fd.read()
            d = {'model_str': model_str}
            return d

        def __setstate__(self, state):
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
                fd.write(state['model_str'])
                fd.flush()
                model = keras.models.load_model(fd.name)
            self.__dict__ = model.__dict__

        cls = keras.models.Model
        cls.__getstate__ = __getstate__
        cls.__setstate__ = __setstate__
