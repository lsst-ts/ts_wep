# This script along with generateTemplates.py
# is to generate the phosim templates contained in the
# policy/cwfs/templateData directory and to be used with
# donutTemplatePhosim.py

import os
import numpy as np
from astropy.io import fits


if __name__ == "__main__":

    ts_wep_dir = os.getenv("TS_WEP_DIR")
    input_repo_dir = os.path.join(
        ts_wep_dir, "policy", "cwfs", "templateData", "tempDir", "input"
    )
    intra_dir = os.path.join(
        input_repo_dir, "rerun", "run1", "postISRCCD", "09006002-g"
    )
    extra_dir = os.path.join(
        input_repo_dir, "rerun", "run1", "postISRCCD", "09006001-g"
    )
    phosim_template_dir = os.path.join(
        ts_wep_dir, "policy", "cwfs", "templateData", "phosimTemplates"
    )
    phosim_output_dir = os.path.join(
        ts_wep_dir, "policy", "cwfs", "templateData", "tempDir", "phosimOutput"
    )

    # This will generate templates 240 x 240 pixels. Should be large enough for any needs.
    stamp_half_width = 120

    print("Generating intra templates")
    for sensor_dir in os.listdir(intra_dir):
        intra_sensor_path = os.path.join(intra_dir, sensor_dir)
        for template_file in os.listdir(intra_sensor_path):
            # Open ISR File
            test_hdu = fits.open(os.path.join(intra_sensor_path, template_file))
            split_name = template_file.split("-")
            template_name = "intra_template-%s_%s.txt" % (split_name[2], split_name[3])

            # Pick an area around the phosim centroid as the template
            centroid_filename = "centroid_lsst_e_9006002_f1_%s_%s_E000.txt" % (
                split_name[2],
                split_name[3],
            )
            centroid_data = np.genfromtxt(
                os.path.join(phosim_output_dir, "intra", centroid_filename),
                unpack=True,
                skip_header=1,
            )
            centroid_x = int(centroid_data[2])
            centroid_y = int(centroid_data[3])
            template_stamp = test_hdu[1].data[
                centroid_x - stamp_half_width : centroid_x + stamp_half_width,
                centroid_y - stamp_half_width : centroid_y + stamp_half_width,
            ]
            # Reduce background noise
            template_stamp[template_stamp < 50] = 0.0
            template_stamp[template_stamp > 0.5] = 1.0
            template_stamp_binary = np.array(template_stamp, dtype=np.int)
            # Save to file
            np.savetxt(
                os.path.join(phosim_template_dir, "%s" % template_name),
                template_stamp,
                fmt="%i",
            )

    print("Generating extra templates")
    for sensor_dir in os.listdir(extra_dir):
        extra_sensor_path = os.path.join(extra_dir, sensor_dir)
        for template_file in os.listdir(extra_sensor_path):

            # Open ISR File
            test_hdu = fits.open(os.path.join(extra_sensor_path, template_file))
            split_name = template_file.split("-")
            template_name = "extra_template-%s_%s.txt" % (split_name[2], split_name[3])

            # Pick an area around the phosim centroid as the template
            centroid_filename = "centroid_lsst_e_9006001_f1_%s_%s_E000.txt" % (
                split_name[2],
                split_name[3],
            )
            centroid_data = np.genfromtxt(
                os.path.join(phosim_output_dir, "extra", centroid_filename),
                unpack=True,
                skip_header=1,
            )
            centroid_x = int(centroid_data[2])
            centroid_y = int(centroid_data[3])
            template_stamp = test_hdu[1].data[
                centroid_x - stamp_half_width : centroid_x + stamp_half_width,
                centroid_y - stamp_half_width : centroid_y + stamp_half_width,
            ]
            # Turn into binary image
            template_stamp[template_stamp < 50] = 0.0
            template_stamp[template_stamp > 0.5] = 1.0
            template_stamp_binary = np.array(template_stamp, dtype=np.int)
            # Save to file
            np.savetxt(
                os.path.join(phosim_template_dir, "%s" % template_name),
                template_stamp,
                fmt="%i",
            )
