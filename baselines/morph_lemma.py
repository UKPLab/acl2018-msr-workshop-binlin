import os
import random
import sys

random.seed(1)

sys.path.append('../../sr2018')
import logging
import sys

from components.utils.readers import ConllFileReader
from components.utils.readers import raw_data_to_tokens

def main():
    logging.basicConfig(level=logging.DEBUG)

    dev_conll_dir = os.path.abspath(sys.argv[1])
    logging.info('CoNLL files dir: %s', dev_conll_dir)

    output_dir = os.path.abspath(sys.argv[2])
    hypotheses_dir = os.path.join(output_dir, 'hyp')
    references_dir = os.path.join(output_dir, 'ref')

    if not os.path.exists(hypotheses_dir):
        os.makedirs(hypotheses_dir)

    if not os.path.exists(references_dir):
        os.makedirs(references_dir)

    for filename in os.listdir(dev_conll_dir):

        conll_data_fname = os.path.join(dev_conll_dir, filename)
        ref_filename = os.path.join(references_dir, filename)
        out_filename = os.path.join(hypotheses_dir, filename)

        raw_data = ConllFileReader.read_file(conll_data_fname)
        tokenized_data = raw_data_to_tokens(raw_data)

        with open(ref_filename, 'w') as reffh, open(out_filename, 'w') as hypfh:

            for idx, instance in enumerate(tokenized_data):
                for t in instance:
                    form = t.GOLD_FORM
                    lemma = t.LEMMA

                    reffh.write('%s\n' % form)
                    hypfh.write('%s\n' % lemma)

    logging.info('Done')

if __name__ == '__main__':
    main()
