import tkinter as tk
import numpy as np
import threading
import time


VALUE_MARKER_SIZE = 10
WINDOW_WIDTH = 400
MAX_RECENT_STEPS = 100
STEP_MIN_PAUSE = 0.05


def val_to_col(v):
    value = int(250 - (215*v))
    return '#%02x%02x%02x' % (value, value, value)


def vec_to_panels(parent, vec):
    canvas = tk.Canvas(parent, width=VALUE_MARKER_SIZE, height=(len(vec)+1)*VALUE_MARKER_SIZE)
    for i in range(len(vec)):
        col = val_to_col(vec[i])
        y = VALUE_MARKER_SIZE*i
        canvas.create_rectangle(0, y, VALUE_MARKER_SIZE, y+VALUE_MARKER_SIZE, fill=col)
    canvas.pack(side='left', anchor='w')
    return canvas


class StateFrame(tk.Frame):
    def __init__(self, num_actions, num_words, num_read_heads, num_write_heads, parent=None):
        super().__init__(parent)
        self.pack(anchor='w')
        self._parent = parent
        self._num_actions = num_actions
        self._num_words = num_words
        self._num_read_heads = num_read_heads
        self._num_write_heads = num_write_heads
        self._actions_frame = tk.LabelFrame(self, text='Actions')
        self._actions_frame.pack(side='top', anchor='w', fill='y')
        self._action_recent = []
        self._read_weights_frames = []
        self._read_weights_recent = []
        for i in range(num_read_heads):
            rwf = tk.LabelFrame(self, text='Read weights '+str(i))
            rwf.pack(side='top', anchor='w', fill='y')
            self._read_weights_frames.append(rwf)
            self._read_weights_recent.append([])
        self._write_weights_frames = []
        self._write_weights_recent = []
        for i in range(num_write_heads):
            wwf = tk.LabelFrame(self, text='Write weights '+str(i))
            wwf.pack(side='top', anchor='w', fill='y')
            self._write_weights_frames.append(wwf)
            self._write_weights_recent.append([])
        self._add_zero_init()

    def _add_zero_init(self):
        self._append(self._action_recent, self._actions_frame, np.zeros(self._num_actions))
        for i in range(len(self._read_weights_frames)):
            self._append(self._read_weights_recent[i], self._read_weights_frames[i], np.zeros(self._num_words))
        for i in range(len(self._write_weights_frames)):
            self._append(self._write_weights_recent[i], self._write_weights_frames[i], np.zeros(self._num_words))

    def _append(self, lst, frame, vec):
        lst.append(vec_to_panels(frame, vec))
        while len(lst) > MAX_RECENT_STEPS:
            lst.pop(0).destroy()

    def add_action_vector(self, action_index):
        a = np.zeros(self._num_actions)
        a[action_index] = 1
        self._append(self._action_recent, self._actions_frame, a)
        return self._parent.winfo_width()

    def add_weights(self, read_weights, write_weights):
        """ expecting (num_X_heads, num_words) shaped vectors for both """
        for i in range(len(self._read_weights_frames)):
            self._append(self._read_weights_recent[i], self._read_weights_frames[i], read_weights[i])
        for i in range(len(self._write_weights_frames)):
            self._append(self._write_weights_recent[i], self._write_weights_frames[i], write_weights[i])


class DNCVisualizedPlayer(tk.Frame):
    @staticmethod
    def player(env, model, nstack=4):
        root = tk.Tk()
        root.wm_title("DNC Player")
        app = DNCVisualizedPlayer(env, model, nstack=nstack, parent=root)
        app.mainloop()
        return app

    def __init__(self, env, model, nstack=4, parent=None):
        super().__init__(parent)
        self.pack()
        self._LOCK = threading.Lock()
        self._env = env
        self._model = model
        self._step_min_pause = STEP_MIN_PAUSE
        self._last_step = time.time()
        self._can_step = True
        self._can_reset = True

        if len(env.observation_space.shape) == 3:
            nh, nw, self._nc = env.observation_space.shape
            self._observation = np.zeros((1, nh, nw, self._nc*nstack), dtype=np.uint8)
            self._update_obs = self._update_obs_3d
        else:
            self._nc = env.observation_space.shape[-1]
            self._observation = np.zeros((1, self._nc*nstack), dtype=np.uint8)
            self._update_obs = self._update_obs_1d
        self._update_obs(self._env.reset())
        self._episode_reward = 0
        self._episode_steps = 0

        self._model_state = model.initial_state
        self._batch_size = len(self._model_state.access_state.read_weights)
        self._num_words = len(self._model_state.access_state.read_weights[0][0])
        self._num_read_heads = len(self._model_state.access_state.read_weights[0])
        self._num_write_heads = len(self._model_state.access_state.write_weights[0])

        self._num_actions = self._env.action_space.n
        self._env.render()
        self._step_button,\
            self._reset_button = self._add_interaction_frame(self)
        self._episode_reward_label,\
            self._episode_steps_label = self._add_info_frame(self)

        parent.bind('s', self.step)
        parent.bind('r', self.reset)

        height = (self._num_write_heads + self._num_read_heads) * (self._num_words+3) + self._num_actions + 3
        height *= VALUE_MARKER_SIZE
        self._canvas = tk.Canvas(self, width=WINDOW_WIDTH, height=height, scrollregion=(0, 0, 0, 0))
        self._canvas.pack()
        self._canvas_scroll = tk.Scrollbar(self, orient=tk.HORIZONTAL, command=self._canvas.xview)
        self._canvas_scroll.pack(fill='x')
        self._canvas.configure(xscrollcommand=self._canvas_scroll.set)
        self._state_frames_container = tk.Frame(self._canvas)
        self._canvas.create_window((0, 0), window=self._state_frames_container, anchor='nw')
        self._state_frames = []
        parent.wm_resizable(width=False, height=False)
        self.reset()

    def _update_obs_3d(self, new_obs):
        self._observation = np.roll(self._observation, shift=-self._nc, axis=3)
        self._observation[:, :, :, -self._nc:] = new_obs

    def _update_obs_1d(self, new_obs):
        self._observation = np.roll(self._observation, shift=-self._nc, axis=1)
        self._observation[:, -self._nc:] = new_obs

    def _update_scrolling(self, width):
        self._canvas.configure(scrollregion=(0, 0, width+10, 0))
        self._canvas.xview_moveto(width+10)

    def _update_info_ui(self):
        self._episode_reward_label.config(text="Episode reward:\t" + str(self._episode_reward))
        self._episode_steps_label.config(text="Episode steps:\t" + str(self._episode_steps))

    def _add_interaction_frame(self, parent):
        interactions_frame = tk.Frame(parent, borderwidth=5)
        interactions_frame.pack(anchor='w')
        reset_button = tk.Button(interactions_frame, text="Reset (R)", command=self.reset)
        reset_button.pack(side='left', anchor='w')
        reset_button.config(state=tk.DISABLED)
        step_button = tk.Button(interactions_frame, text="Step (S)", command=self.step)
        step_button.pack(side='left', anchor='w')
        return step_button, reset_button

    def _add_info_frame(self, parent):
        info_frame = tk.Frame(parent, borderwidth=5)
        info_frame.pack(anchor='w')
        episode_reward_label = tk.Label(info_frame, text="Episode reward:")
        episode_reward_label.pack(side='top', anchor='w')
        episode_steps_label = tk.Label(info_frame, text="Episode steps:")
        episode_steps_label.pack(side='top', anchor='w')
        return episode_reward_label, episode_steps_label

    def reset(self, unused_arg=None):
        self._LOCK.acquire()
        if not self._can_reset:
            self._LOCK.release()
            return
        self._episode_reward = 0
        self._episode_steps = 0
        self._update_obs(self._env.reset())
        for f in self._state_frames:
            f.destroy()
        self._state_frames = []
        for _ in range(self._batch_size):
            self._state_frames.append(StateFrame(self._num_actions,
                                                 self._num_words,
                                                 self._num_read_heads,
                                                 self._num_write_heads,
                                                 self._state_frames_container))
        self._can_step = True
        self._can_reset = False
        self._reset_button.configure(state=tk.DISABLED)
        self._step_button.configure(state=tk.NORMAL)
        self._update_scrolling(0)
        self._update_info_ui()
        self._env.render()
        self._LOCK.release()

    def _add_actions(self, new_actions):
        """ len() = batchsize, each providing an action index """
        width = 0
        for i in range(len(new_actions)):
            width = max([self._state_frames[i].add_action_vector(new_actions[i]), width])
        self._update_scrolling(width)

    def _add_state(self, dnc_state):
        for i in range(self._batch_size):
            rw = dnc_state.access_state.read_weights[i]
            ww = dnc_state.access_state.write_weights[i]
            self._state_frames[i].add_weights(rw, ww)
        pass

    def step(self, render=True):
        if not self._can_step:
            return
        self._LOCK.acquire()
        if (time.time() - self._last_step) < self._step_min_pause:
            self._LOCK.release()
            return False
        self._last_step = time.time()
        new_action, values, self._model_state = self._model.step(self._observation, self._model_state, [False])
        new_obs, reward, done, info = self._env.step(new_action[0])
        self._add_actions(new_action)
        self._add_state(self._model_state)
        self._update_obs(new_obs)
        self._episode_reward += reward
        self._episode_steps += 1
        self._update_info_ui()
        if render:
            self._env.render()
        if done:
            self._can_step = False
            self._can_reset = True
            self._reset_button.config(state=tk.NORMAL)
            self._step_button.config(state=tk.DISABLED)
        self._LOCK.release()
        return done
