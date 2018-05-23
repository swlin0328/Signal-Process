#!/usr/bin/python
# -*- coding: utf-8 -*-
import MySQLdb, cPickle
from time import strftime
from keras.models import load_model
import types
import tempfile
import keras.models

class sql4Keras():
    def __init__(self, model_name, appliance, user="", description=""):
        self.mySQL_connect()
        self.user = user
        self.name = model_name
        self.description = description
        self.appliance = appliance
        self.created_time = strftime('%Y-%m-%d_%H_%M')
        self.chk_model_exist()

    def upload_h5_to_SQL(self, model_path, valid_metrics=[0, 0, 0, 0], step=0):
        from keras.models import load_model
        model = load_model(model_path)

        self.mySQL_connect()
        self.save2sql(model, valid_metrics, step)


    def mySQL_connect(self):
        print('=====================================================')
        print('======== Connect to the remote mySQL server ========')
        print('=====================================================')
        print('Time : {}\n'.format(strftime('%Y-%m-%d_%H_%M')))
        host_cable = "140.115.50.100"
        #host_wifi = "140.115.30.152"

        self.db = MySQLdb.connect(host=host_cable, port=3306, user="NILM", passwd="NILM", db="NILM",
                        charset="utf8")
        self.db.ping(True)
        self.cursor = self.db.cursor()

    def insert_model(self, model, valid_metrics, step):
        add_model = "INSERT INTO Model (name, description, layers, mse, mae, appliance, params, step, created_time, user) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
        self.cursor.execute(add_model, (self.name, self.description, len(model.layers), valid_metrics[3], valid_metrics[2], self.appliance, model.count_params(), step, self.created_time, self.user))
        self.db.commit()

    def search_model(self):
        search_id = "SELECT id FROM Model WHERE name = %s"
        self.cursor.execute(search_id, (self.name,))
        self.model_id = self.cursor.fetchone()

    def insert_layer(self, model):
        self.search_model()

        for idx in range(len(model.layers)):
            layer = model.layers[idx]
            config = layer.get_config()
            model_id = int(self.model_id[0])
            layer_id = idx
            model_name = str(config.get('activation'))
            layer_output = str(layer.output_shape)
            num_params = layer.count_params()
            activation = str(config.get('activation', "---"))

            add_layer = "INSERT INTO Layer (model_id, layer_id, layer_type, output_shape, params, description, created_time) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            self.cursor.execute(add_layer, (model_id, layer_id,  model_name, layer_output, num_params, activation, self.created_time))
            self.db.commit()

    def insert_model_blob(self, trained_model):
        self.search_model()
        self.make_keras_picklable()
        model_blob = cPickle.dumps(trained_model)

        add_blob = "INSERT INTO Model_Blob (model_id, data) VALUES (%s, %s)"
        self.cursor.execute(add_blob, (self.model_id, model_blob))
        self.db.commit()

    def read_model_blob(self):
        print('Fetch the target model...')
        self.search_model()

        sql_cmd = "SELECT data FROM Model_Blob WHERE Model_id = %s"
        self.cursor.execute(sql_cmd, (self.model_id,))
        model_blob = self.cursor.fetchone()
        return model_blob

    def load_model(self):
        self.mySQL_connect()
        self.make_keras_picklable()
        sql_model = self.read_model_blob()
        model = cPickle.loads(sql_model[0])
        return model

    def read_model_info(self):
        self.mySQL_connect()
        self.cursor.execute("SELECT * FROM Model")
        results = self.cursor.fetchall()
        for record in results:
            print(record)

    def disconnect(self):
        self.db.close()
        print('=====================================================')
        print('============ Close the remote connection ============')
        print('=====================================================')

    def save2sql(self, model, valid_metrics, step):
        self.mySQL_connect()
        self.insert_model(model, valid_metrics, step)
        self.insert_layer(model)
        self.insert_model_blob(model)

    def chk_model_exist(self):
        self.search_model()
        if self.model_id != None:
            print('Warning: model_name exist')
            return True
        return False

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
