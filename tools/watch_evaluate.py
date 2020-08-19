import os
import time
import logging

from fire import Fire

from concern.smart_path import smart_path


def main(dest_dir="s3://detection/benchmarking/inference/"):
    dest_dir = smart_path(dest_dir)

    while True:
        for prediction_path in dest_dir.glob("*.pth"):
            command = "rlaunch --cpu 2 --memory $((10*1024)) -- python3 -m " \
                      "tools.result_statistic --config-file configs/retina/base_retina_R_50_FPN_1x.yaml " \
                      "--prediction {}".format(prediction_path.as_uri())
            print(command)
            code = os.system(command)
            print(code)
            if code == 0:
                prediction_path.rename(
                    dest_dir.joinpath("archived", prediction_path.name))
            else:
                print("Error for {}, will retry in 10 secs".format(prediction_path.as_uri()))

        print("Dir {} is clean, sleep for 10 secs.".format(dest_dir.as_uri()))
        time.sleep(10)


Fire(main)
