import os
import pandas as pd
import click
from datetime import datetime
import json
from pathlib import Path


class Experiments():
    
    def __init__(self, folder:str = ''):
        # validate
        if not os.path.exists(folder):
            self.__mkdir(folder)
        # set
        self.folder = folder
        
        
    def __build_name(self, name:str)->str:
        # get number id
        l_exp = self.show()
        if len(l_exp) == 0:
            nid = 1
        else:
            nid = int(l_exp[-1].split('-')[0]) + 1
        # build name and return
        return f'{nid}-{datetime.now().strftime("%Y%m%d%H%M%S")}-{name.replace(" ","_")}.json'
 

    @staticmethod
    def __save_dict_to_json(dictionary:dict, path:str):
        # validation
        assert '.json' in path, 'It is recommended the extension .json in path.'
        # save
        with open(path, "w") as outfile:
            json.dump(dictionary, outfile)

            
    @staticmethod
    def __load_json_to_dict(path:str):
        # validation
        assert os.path.isfile(path), f'The file "{path}" not exists.'
        # load
        with open(path, 'r') as openfile:
            dict_object = json.load(openfile)
        # return
        return dict_object


    @staticmethod
    def __mkdir(folder:str):    
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        click.secho(f'[info] It was created the folder "{click.format_filename(folder, shorten=True)}".', fg='green')


    def show(self):
        return sorted([file for file in os.listdir(self.folder) if file[-5:] == '.json'])
    
    
    def save(self, name:str, comment:str = '', metrics:dict = {}, hyperparameters:dict = {}, features_names:list = [], model:"estimator" = None):
        # validate arguments
        assert type(metrics) == dict, click.secho('[error] Argument "metrics" must be a dictionary.', fg='red')
        assert type(hyperparameters) == dict, click.secho('[error] Argument "hyperparameters" must be a dictionary.', fg='red')
        assert type(features_names) == list, click.secho('[error] Argument "features_names" must be a dictionary.', fg='red')
        if len(metrics) == 0:
            click.secho('[warning] Argument "metrics" was not passed.', fg='yellow')
        # validate if name already exits
        l_names = self.load().name.tolist()
        if name in l_names:
            click.secho('[warning] This name already exits: It will not saved this experiment.', fg='yellow')
        else:
            # build file name
            file_name = self.__build_name(name)
            # build dict to be saved
            dcontent = {
                'name': name,
                'comment': comment
                }
            # include metrics
            if len(metrics) > 0:
                dcontent['metrics'] = metrics
            # include hyperparamenters
            if len(hyperparameters) > 0:
                dcontent['hyperparameters'] = hyperparameters           
            # include features_names
            if len(features_names) > 0:
                dcontent['features_names'] = features_names   
            # save dict in json
            self.__save_dict_to_json(dcontent, os.path.join(self.folder,file_name))
        
        
    def clean(self):
        for file in self.show():
            rem_path = os.path.join(self.folder, file)
            rem_file = Path(rem_path)
            rem_file.unlink()
            
            
    def load(self, nid:int = None):
        # collect list of files
        files = self.show()
        # validate if it is required just one experiment by id
        if not nid is None:
            # collect select file name by id
            l_file_x = [file for file in files if int(file.split('-')[0]) == nid]
            # validate if exits
            if len(l_file_x) == 0:
                click.secho(f'[error] There are not any experiment with id "{nid}".', fg='red')
            else:
                # load and return
                return self.__load_json_to_dict(os.path.join(self.folder, l_file_x[0]))
        else:
            # validate if there are some experiment
            if len(files) == 0:
                return pd.DataFrame({'nid':[], 'creation_dt':[], 'name':[], 'comment':[]})
            # loop of files
            for ii, file in enumerate(files[:]):
                # parse file name
                nid, dt = file.split('-')[:2]
                nid = int(nid)
                dt = datetime.strptime(dt, '%Y%m%d%H%M%S')
                # load json to dict
                dfile = self.__load_json_to_dict(os.path.join(self.folder, file))
                # dict to df
                df = pd.DataFrame({'nid':nid, 'creation_dt':[dt], 'name':[dfile['name']], 'comment':[dfile['comment']]})
                # add metrics if exits
                if 'metrics' in list(dfile.keys()):
                    for k,v in dfile['metrics'].items():
                        df[f'metric_{k}'] = [v]
                # add hyperparameter if exits
                if 'hyperparameters' in list(dfile.keys()):
                    for k,v in dfile['hyperparameters'].items():
                        df[f'hp_{k}'] = [v]
                # append records
                if ii == 0:
                    dfexp = df.copy()
                else:
                    dfexp = pd.concat([dfexp, df], axis = 0)
            # reset index
            dfexp.reset_index(drop = True, inplace = True)
            # return
            return dfexp.set_index('nid')