

import logging
import os
import time

import torch
import torch.nn as nn

from components.constants import PAD_ID
from components.trainer.base_trainer import BaseTrainer
from components.utils.readers import conll2snt
from components.utils.readers import read_conll_data_file
from components.utils.serialization import save_txt
from components.utils.timing import asMinutes

logger = logging.getLogger('main')

class SRMLPTrainer(BaseTrainer):

    def training_start(self, model, data, evaluator, nlgen):

        logger.debug("Preparing training data")

        dev_data_fname = data.fnames.dev_fn
        assert os.path.exists(dev_data_fname), logger.error('File %s does not exist', dev_data_fname)

        # dev data for evaluation
        dev_data_ref_fname = data.fnames.dev_ref_fn
        dev_data_raw = read_conll_data_file(data.fnames.dev_fn)
        logger.info('Saving Syn reference --> %s', data.fnames.dev_ref_fn)
        save_txt(itemlist=conll2snt(dev_data_raw), fname=dev_data_ref_fname)

        train_batches = data.batchify_vectorized_data(data.train, self.batch_size) # [(np_x, np_y_1hot), ...]
        dev_batches = data.batchify_vectorized_data(data.dev, self.batch_size)

        # need to move the model before setting the optimizer
        # see: http://pytorch.org/docs/stable/optim.html
        self.set_optimizer(model, self.config['optimizer'])
        self.set_train_criterion(len(data.vocab.id2tok), PAD_ID)

        training_start_time = time.time()
        logger.info("Start training")

        best_model_fn = None
        best_weights = None

        best_score = 0
        best_loss = None
        dev_losses = []

        for epoch_idx in range(1, self.n_epochs + 1):
            epoch_start = time.time()

            # compute loss on train and dev data
            train_loss = self.train_epoch(epoch_idx, model, train_batches)
            dev_loss = self.compute_val_loss(model, dev_batches)
            evaluator.record_loss(train_loss, dev_loss)

            # # run on dev data in prediction mode (no oracle decoding)
            # predictions_fname = self.get_predictions_fname(epoch_idx)
            # depgraphs = nlgen.predict_from_raw_data(model, dev_data_raw, data.vocab)
            # nlgen.save_predictions(depgraphs, predictions_fname)

            # # evaluate using metrics
            # scores = evaluator.external_metric_eval(ref_fn=dev_data_ref_fname, pred_fn=predictions_fname)
            # avg_score = (scores.bleu + scores.edist) / 2
            # model_fn = os.path.join(self.model_dir,
            #                         'weights.epoch%d_%0.3f_%0.3f' % (epoch_idx, scores.bleu, scores.edist))

            # if avg_score > best_score:
            #     best_score = avg_score
            #     best_model_fn = model_fn
            #     best_weights = model.state_dict()
            if best_loss is None or dev_loss < best_loss:
                best_loss = dev_loss
                best_weights = model.state_dict()

            epoch_time = asMinutes(time.time() - epoch_start)
            logger.info('Epoch %d/%d: tloss=%0.4f dloss=%0.4f time = %s',
                        epoch_idx, self.n_epochs, train_loss, dev_loss, epoch_time)

        logger.info('Total training time=%s' % (asMinutes(time.time() - training_start_time)))

        self.best_model_fn = os.path.join(self.model_dir, 'model.pt')
        logger.debug('Saving model to --> %s', self.best_model_fn)
        torch.save(best_weights, self.best_model_fn)

        # score_fname = os.path.join(self.model_dir, 'scores.csv')
        # scores = evaluator.get_scores_to_save()
        # evaluator.save_scores(scores, self.score_file_header, score_fname)
        #
        # evaluator.plot_lcurve(fname = os.path.join(self.model_dir, "lcurve.pdf"),
        #                       title= self.model_type)
        #
        # evaluator.save_errors(errors=nlgen.error_analysis_data,
        #                       fname=os.path.join(self.model_dir, 'syn.dev.error-depgraphs.pkl'))

        # restore best weights of the model
        model.load_state_dict(best_weights)
        model.eval()
        dev_predictions = nlgen.predict_from_file(model, data.config['dev_data'], data.vocab)
        predictions_fname = '%s.out.dev' % self.best_model_fn
        reference_fname = '%s.ref' % predictions_fname
        nlgen.save_predictions(dev_predictions, predictions_fname)
        nlgen.save_dev_references(reference_fname)

        srdev_predictions = nlgen.predict_from_file(model, data.config['test_data'], data.vocab)
        predictions_fname = '%s.out.srdev' % self.best_model_fn
        nlgen.save_predictions(srdev_predictions, predictions_fname)
        # evaluator.external_metric_eval(ref_fn=reference_fname, pred_fn=predictions_fname)

        return

    def set_train_criterion(self, vocab_size, pad_id):

        # if weighting scheme:
        # weight = torch.ones(vocab_size)
        # weight[pad_id] = 0
        self.criterion = nn.BCELoss(size_average=True)
        self.criterion.to(self.device)

    def train_step(self, model, batch):
        logits = model.forward(batch[0])  #
        loss_var = self.calc_loss(logits, batch[1])
        return loss_var

    def calc_loss(self, logits, y_var):
        return self.criterion(logits, y_var)

    @property
    def score_file_header(self):
        HEADER = ['bleu', 'nist', 'edist', 'train_loss', 'dev_loss']
        return HEADER

component = SRMLPTrainer
