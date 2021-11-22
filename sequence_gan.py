import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import pickle
from tqdm import tqdm 

tf.compat.v1.disable_eager_execution()

###################################################################################
#  Generator  Hyper-parameters
###################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 20 #150 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64

###################################################################################
#  Discriminator  Hyper-parameters
###################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

###################################################################################
#  Basic Training Parameters
###################################################################################
TOTAL_BATCH = 10 #200
positive_file = 'data/real_hittype.txt'
negative_file = 'data/generator_sample.txt'
eval_file = 'data/eval_hittype.txt'
generated_num = 4730 #10000 # same as train_data length

log_file = open('logs/train.log', 'w')

def log(value):
    s = str(value) 
    log_file.write(s)


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def gen_eval(sess, generator, eval_loader):
    eval_loader.reset_pointer()
    eval_losses = []

    for it in range(eval_loader.num_batch):
        batch = eval_loader.next_batch()
        loss = sess.run(generator.pretrain_loss, {generator.x: batch})
        eval_losses.append(loss)

    return np.mean(eval_losses)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    vocab_size = 12
    
    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    eval_loader = Gen_Data_loader(BATCH_SIZE)
    #likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing
    dis_data_loader = Dis_dataloader(BATCH_SIZE)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    #target_params = pickle.load(open('save/target_params.pkl', 'rb'), encoding='latin1')
    #target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model

    discriminator = Discriminator(sequence_length=20, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())

    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    #generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
    gen_data_loader.create_batches(positive_file)
    eval_loader.create_batches(eval_file)

    #  pre-train generator
    print('Start pre-training...')
    log('pre-training...\n')
    for epoch in range(PRE_EPOCH_NUM):
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        buffer = '[{}/{}] Train loss {:.4f}'.format(epoch,PRE_EPOCH_NUM,loss)
        print(buffer)
        log(buffer)

        if epoch % 5 == 0:    
            test_loss = gen_eval(sess, generator, eval_loader)
            #generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            #likelihood_data_loader.create_batches(eval_file)
            #test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print('pre-train epoch ', epoch, 'test_loss ', test_loss)
            #buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
            #log(buffer)

    exit(0)

    print('Start pre-training discriminator...')
    # Train 3 epoch on the generated data and do this for 50 times
    for _ in tqdm(range(50)):
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _ = sess.run(discriminator.train_op, feed)


    rollout = ROLLOUT(generator, 0.8)

    print('######################################################################')
    print('Start Adversarial Training...')
    log('adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        for it in range(1):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)

        # Test
        #if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            #generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            #likelihood_data_loader.create_batches(eval_file)
            #test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            #buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
            #print('total_batch: ', total_batch, 'test_loss: ', test_loss)
            #log(buffer)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        for _ in range(5):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)

            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op, feed)

    log_file.close()


if __name__ == '__main__':
    main()
