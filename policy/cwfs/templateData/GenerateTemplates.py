# This script along with generateTemplates.py
# is to generate the phosim templates contained in the
# policy/cwfs/templateData directory and to be used with
# donutTemplatePhosim.py

import os
import numpy as np
from astropy.io import fits
from scipy.signal import correlate


if __name__ == "__main__":

    ts_wep_dir = os.getenv('TS_WEP_DIR')
    input_repo_dir = os.path.join(ts_wep_dir, 'policy', 'cwfs', 'templateData', 'tempDir', 'input')
    intra_dir = os.path.join(input_repo_dir, 'rerun', 'run1', 'postISRCCD', '09006001-g', 'R22')
    extra_dir = os.path.join(input_repo_dir, 'rerun', 'run1', 'postISRCCD', '09006002-g', 'R22')
    phosim_template_dir = os.path.join(ts_wep_dir, 'policy', 'cwfs', 'templateData', 'phosimTemplates')

    # Pick an initial image with a donut to use at a convolution template for centroiding
    template_match = fits.open(os.path.join(intra_dir, 'postISRCCD_09006001-g-R22-S00-det090.fits'))
    template_match_stamp = template_match[1].data[2080:2240, 2105:2265]

    # This should generate templates large enough for any needs
    stamp_half_width = 120

    print('Generating intra templates')
    for template_file in os.listdir(intra_dir):
        test_hdu = fits.open(os.path.join(intra_dir, template_file))
        split_name = template_file.split('-')
        template_name = 'intra_template-%s_%s.txt' % (split_name[2], split_name[3])
        correlated_stamp = correlate(test_hdu[1].data, template_match_stamp, mode='same')
        correlate_max = np.argmax(correlated_stamp)
        correlate_coord = np.unravel_index(correlate_max, np.shape(template_match[1].data))
        final_template = test_hdu[1].data[correlate_coord[0]-stamp_half_width:correlate_coord[0]+stamp_half_width,
                                          correlate_coord[1]-stamp_half_width:correlate_coord[1]+stamp_half_width]
        np.savetxt(os.path.join(phosim_template_dir, '%s' % template_name), final_template)

    print('Generating extra templates')
    for template_file in os.listdir(extra_dir):
        test_hdu = fits.open(os.path.join(extra_dir, template_file))
        split_name = template_file.split('-')
        template_name = 'extra_template-%s_%s.txt' % (split_name[2], split_name[3])
        correlated_stamp = correlate(test_hdu[1].data, template_match_stamp, mode='same')
        correlate_max = np.argmax(correlated_stamp)
        correlate_coord = np.unravel_index(correlate_max, np.shape(template_match[1].data))
        final_template = test_hdu[1].data[correlate_coord[0]-stamp_half_width:correlate_coord[0]+stamp_half_width,
                                          correlate_coord[1]-stamp_half_width:correlate_coord[1]+stamp_half_width]
        np.savetxt(os.path.join(phosim_template_dir, '%s' % template_name), final_template)
