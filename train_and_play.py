from global_params import*
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import regularizers
from snapshots_env import*
from mcts_nodes import*
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
tf.disable_v2_behavior()


def create_model_tensors():
    state_t = tf.placeholder('float32', [None,] + list(state_dim),name='state_t')
    action_t = tf.placeholder(tf.float32, shape=[None, n_actions], name='action_t')
    value_t = tf.placeholder(tf.float32, shape=[None, 1], name='value_t')

    x = Conv2D(16, [3, 3], strides=[2, 2, ], activation='relu', kernel_regularizer=regularizers.l2(c))(state_t)
    x = Conv2D(32, [3, 3],strides=[2, 2, ], activation='relu', kernel_regularizer=regularizers.l2(c))(x)
    x = Conv2D(64, [3, 3],strides=[2, 2, ],activation='relu', kernel_regularizer=regularizers.l2(c))(x)
    x = Flatten()(x)
    x1 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(c))(x)
    x2 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(c))(x)

    policy_output_layer = Dense(n_actions, activation='softmax', kernel_regularizer=regularizers.l2(c))(x1)
    value_output_layer = Dense(1, activation='linear', kernel_regularizer=regularizers.l2(c))(x2)

    l1 = tf.reduce_mean((value_output_layer - value_t) ** 2)
    l2 = tf.reduce_mean(categorical_crossentropy(action_t, policy_output_layer))

    total_loss = l1 + l2
    train_step = tf.train.AdamOptimizer(adam_lr).minimize(total_loss)
    return state_t, action_t, value_t, policy_output_layer, value_output_layer, train_step, total_loss


def play_game(cur_game_num, sess, policy_output_layer, value_output_layer, state_t):

    env = WithSnapshots(gym.make("Breakout-ram-v0").env)
    root_observation = env.reset()
    root_snapshot = env.get_snapshot()
    root = Root(root_snapshot, root_observation, env)

    train_pics = np.zeros((max_number_of_rounds_per_game,) + state_dim)
    train_actions = np.zeros((max_number_of_rounds_per_game, n_actions))
    train_value = np.zeros((max_number_of_rounds_per_game, 1))
    train_action_value = np.zeros((max_number_of_rounds_per_game, 1))
    rewards_array = np.zeros(max_number_of_rounds_per_game)

    done = 0
    total_reward = 0
    last_lives_changed = 0
    cur_lives = env.unwrapped.ale.lives()
    t_counter = 0

    while not done:

        if t_counter > max_number_of_rounds_per_game:
            break

        # PLANNING MOVE:
        plan_mcts(root, sess, policy_output_layer, value_output_layer, state_t, planning_depth)
        env.restore_snapshot(root.snapshot)

        # CHECK IF DONE:
        if root.is_done:
            print("Finished with reward = ", total_reward)
            print("Finished with lives = ", env.unwrapped.ale.lives())
            root.pop_pic()
            break

        # PLAY BEST MOVE:
        action, best_child, p_choose, action_value = root.play_move(t_counter)
        new_s, r, done, info = env.step(action)  # if default action then every child has is_done marked true
        # so we'll end the loop right here, and not add it as a training example

        if done:
            print("Finished with reward = ", total_reward)
            print("Finished with lives = ", env.unwrapped.ale.lives())
            break

        # ADD DATA TO REPLAY BUFFER:
        env.render()
        total_reward += r
        rewards_array[t_counter] = r
        if cur_lives != env.unwrapped.ale.lives():
            last_lives_changed = t_counter
            cur_lives = env.unwrapped.ale.lives()

        if r != 0:
            train_value[last_lives_changed:t_counter, 0] = 1

        train_pics[t_counter] = format_pics(get_pics(best_child.parent, env))
        train_actions[t_counter] = p_choose
        train_action_value[t_counter] = action_value
        t_counter += 1
        print('game: ' + str(cur_game_num))
        print('action: ' + str(action))
        print('reward: ' + str(r))
        print('step: ' + str(t_counter))
        root = Root.from_node(best_child)

    del env.pics
    env.close()
    train_pics = train_pics[:t_counter]
    train_actions = train_actions[:t_counter]
    train_value = train_value[:t_counter]
    train_action_value = train_action_value[:t_counter]
    return train_pics, train_actions, train_value, train_action_value, total_reward


def play_games_and_save_data(gen_num, sess, state_t, policy_output_layer, value_output_layer):
    game_start = 0
    avg_reward = 0
    for k in range(game_start, num_games_to_play):
        train_pics, train_actions, train_value, train_action_value, total_reward = play_game(k, sess,
                                                                                             policy_output_layer,
                                                                                             value_output_layer,
                                                                                             state_t)

        np.save(generation_folder_path + '\\' + str(gen_num) + 'train_pics_alphazero' + str(k), train_pics)
        np.save(generation_folder_path + '\\' + str(gen_num) + 'train_value_alpazero' + str(k), train_value)
        np.save(generation_folder_path + '\\' + str(gen_num) + 'train_actions_alpazero' + str(k), train_actions)
        np.save(generation_folder_path + '\\' + str(gen_num) + 'train_action_value_alpazero' + str(k), train_action_value)
        clear_output()
        avg_reward += total_reward

    avg_reward = avg_reward / (num_games_to_play - game_start)
    return avg_reward


def load_from_cur_gen_and_prev(gen_num):

    train_pics = np.zeros((100000,) + state_dim)
    train_actions = np.zeros((100000, n_actions))
    train_value = np.zeros((100000, 1))
    train_action_value = np.zeros((100000, 1))
    t_count = 0
    for k in tqdm(range(0, num_games_to_play)):
        cur_actions = np.load(generation_folder_path + '\\' + str(gen_num) + 'train_actions_alpazero' + str(k) + '.npy')
        cur_pics = np.load(generation_folder_path + '\\' + str(gen_num) + 'train_pics_alphazero' + str(k) + '.npy')
        cur_value = np.load(generation_folder_path + '\\' + str(gen_num) + 'train_value_alpazero' + str(k) + '.npy')
        cur_action_value = np.load(generation_folder_path + '\\' + str(gen_num) + 'train_action_value_alpazero' + str(k)
                                   + '.npy')

        train_pics[t_count:(t_count + len(cur_pics))] = cur_pics
        train_actions[t_count:(t_count + len(cur_actions))] = cur_actions
        train_value[t_count:(t_count + len(cur_actions))] = cur_value
        train_action_value[t_count:(t_count + len(cur_actions))] = cur_action_value
        t_count += len(cur_pics)

    # "Sliding window"
    if gen_num > 1:
        for k in tqdm(range(0, num_games_to_play)):
            cur_actions = np.load(generation_folder_path + '\\' + str(gen_num - 1) + 'train_actions_alpazero' + str(k) +
                                  '.npy')
            cur_pics = np.load(generation_folder_path + '\\' + str(gen_num - 1) + 'train_pics_alphazero' + str(k) +
                               '.npy')
            cur_value = np.load(generation_folder_path + '\\' + str(gen_num - 1) + 'train_value_alpazero' + str(k) +
                                '.npy')
            cur_action_value = np.load(generation_folder_path + '\\' + str(gen_num - 1) + 'train_action_value_alpazero'
                                       + str(k) + '.npy')

            train_pics[t_count:(t_count + len(cur_pics))] = cur_pics
            train_actions[t_count:(t_count + len(cur_actions))] = cur_actions
            train_value[t_count:(t_count + len(cur_actions))] = cur_value
            train_action_value[t_count:(t_count + len(cur_actions))] = cur_action_value
            t_count += len(cur_pics)

    train_pics = train_pics[:t_count]
    train_actions = train_actions[:t_count]
    train_value = train_value[:t_count]
    train_action_value = train_action_value[:t_count]
    return train_pics, train_actions, train_value, train_action_value


def play_and_train_generations(start_generation):

    sess = tf.InteractiveSession()
    state_t, action_t, value_t, policy_output_layer, value_output_layer, train_step, total_loss = create_model_tensors()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    avg_rewards = []
    for cur_gen in range(start_generation, num_generations):

        avg_reward = play_games_and_save_data(cur_gen, sess, state_t, policy_output_layer, value_output_layer)
        if cur_gen >= 1:
            saver.restore(sess, model_folder_path + "/model_gen_" + str(cur_gen - 1) + ".ckpt")

        avg_rewards.append(avg_reward)
        print('Finished generation with average reward: ' + str(avg_reward))

        # loading data:
        print('loading data....')
        train_pics, train_actions, train_value, train_action_value = load_from_cur_gen_and_prev(cur_gen)
        if is_to_augment_data:
            new_train_pics, new_train_actions, new_train_value, new_train_action_value = flip_horizontal(
                train_pics, train_actions, train_value, train_action_value)

            train_pics = np.concatenate((train_pics, new_train_pics))
            train_actions = np.concatenate((train_actions, new_train_actions))
            train_value = np.concatenate((train_value, new_train_value))
            train_action_value = np.concatenate((train_action_value, new_train_action_value))

        print('Loading Finished...')
        print('training size: ' + str(len(train_pics)))

        # splitting data:
        losses = []
        val_losses = []
        train_pics, val_pics, train_actions, val_actions, train_value, val_value = train_test_split(train_pics,
                                                                                                    train_actions,
                                                                                                    train_value,
                                                                                                    test_size=test_size)

        # Training agent:

        print('Starting training loop for generation ' + str(cur_gen) + '....')
        for k in range(num_epochs):
            idxs = np.random.choice(len(train_pics), batch_size)
            x = train_pics[idxs]
            y = train_actions[idxs]
            z = train_value[idxs]
            _, curr_loss = sess.run([train_step, total_loss], feed_dict={state_t: x, action_t: y, value_t: z})
            losses.append(curr_loss)
            val_loss = sess.run([total_loss], feed_dict={state_t: val_pics, action_t: val_actions, value_t: val_value})
            val_losses.append(val_loss)
            print('epoch: ' + str(k) + ' curr_loss: ' + str(curr_loss) + " val loss: " + str(val_loss))

        print('training complete')

        if is_to_plot:
            plt.figure()
            plt.plot(losses)
            plt.plot(val_losses)
            plt.savefig(save_res_dir+'/training_gen'+str(cur_gen)+'.png')
            plt.close()

        save_path = saver.save(sess, model_folder_path + "/model_gen_" + str(cur_gen) + ".ckpt")


play_and_train_generations(0)





