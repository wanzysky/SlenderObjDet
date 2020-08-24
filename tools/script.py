import os
import time
import datetime


class Runner(object):

    def __init__(self, model_name, backbones, cfg_files, file='train_net_local.py'):
        assert len(cfg_files) == len(backbones)
        self.model_name = model_name
        self.backbones = backbones
        self.cfg_files = cfg_files
        self.file = file

        self.command_dicts = self.get_commands()

    def run(self):
        for command_dict in self.command_dicts:
            print(command_dict)
            while True:
                ret = os.system(command_dict['command'])
                if ret == 0:
                    break
                else:
                    print(ret)
                    print("Error for {}, will retry in 10 secs".format(command_dict['command']))
                    time.sleep(10)

    def get_commands(self):
        gpu = 8
        cpu = 16
        memory = 80 * 1024
        model_name = self.model_name
        file = self.file

        command_dicts = []
        for backbone, cfg_file in zip(self.backbones, self.cfg_files):
            output_dir = self.get_output_dir(model_name, backbone, schedule='1x')
            command = "rlaunch --gpu {} --cpu {} --memory {} -- python {} " \
                      "--num-gpus {} --resume --config-file {} " \
                      "OUTPUT_DIR {}".format(
                gpu, cpu, memory, file, gpu, cfg_file, output_dir)
            command_dicts.append(
                dict(
                    command=command,
                    info=dict(
                        model_name=model_name,
                        gpu=gpu,
                        cpu=cpu,
                        memory=memory // 1024,
                        output_dir=output_dir,
                    )
                )
            )
        return command_dicts

    @classmethod
    def get_output_dir(cls, model_name, backbone_name, schedule):
        output_dir = 'output/'
        # get data
        today = datetime.date.today()
        month = today.month
        day = today.day
        output_dir += "{:02d}-{:02d}".format(month, day)

        # get model_name
        output_dir += "_{}".format(model_name)

        # get backbone name
        output_dir += "_{}".format(backbone_name)

        # get schedule
        output_dir += "_{}".format(schedule)

        return output_dir


def run_fcos(depths=(18, 34, 50, 101), file='train_net_local.py'):
    model_name = 'fcos'
    backbones = ['R_{}_FPN'.format(depth) for depth in depths]
    cfg_files = [
        'configs/group_exp_for_backbone/fcos/r-{}.yaml'.format(depth) for depth in depths
    ]

    runner = Runner(model_name, backbones, cfg_files, file)
    runner.run()


def run_retina(depths=(18, 34, 50, 101), file='train_net_local.py'):
    model_name = 'retina'
    backbones = ['R_{}_FPN'.format(depth) for depth in depths]
    cfg_files = [
        'configs/group_exp_for_backbone/retina/r-{}.yaml'.format(depth) for depth in depths
    ]

    runner = Runner(model_name, backbones, cfg_files, file)
    runner.run()


def run_faster(depths=(18, 34, 50, 101), file='train_net_local.py'):
    model_name = 'faster'
    backbones = ['R_{}_FPN'.format(depth) for depth in depths]
    cfg_files = [
        'configs/group_exp_for_backbone/faster/r-{}.yaml'.format(depth) for depth in depths
    ]

    runner = Runner(model_name, backbones, cfg_files, file)
    runner.run()


def run_reppoints(depths=(18, 34, 50, 101), file='train_net_local.py'):
    model_name = 'rep-points'
    backbones = ['R_{}_FPN'.format(depth) for depth in depths]
    cfg_files = [
        'configs/group_exp_for_backbone/rep-points/r-{}.yaml'.format(depth) for depth in depths
    ]

    runner = Runner(model_name, backbones, cfg_files, file)
    runner.run()


if __name__ == '__main__':
    import fire

    fire.Fire()
