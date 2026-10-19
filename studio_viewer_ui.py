"""Window-free decisions for the 3D viewer's controls (studio_brain3d.py): colours, which controls show, their labels and
layout, where the block labels go and how the camera frames each level. Nothing here imports Panda3D, so tests can use it.
Sizes are aspect2d units (the window is 2 units tall); control frames are in text units (multiples of TEXT)."""
from __future__ import annotations


# The viewer's colours (the web UI's theme tokens; studio_scene_json.THEME keeps its subset equal to these)
C = dict(
    bg="#0e1014", panel="#15181e", panel2="#1b1f27", border="#2a2f3a", border2="#20242c",
    text="#e6e8ec", text2="#c3c8d1", muted="#8f97a6", dim="#7f879a",
    violet="#9f8cff", amber="#e5b567", coral="#ef8a7a", blue="#6cb4ee", pink="#e58fb0", sage="#9fd36b",
    amber_bg="#211c12", coral_bg="#26171a", blue_bg="#121a22", violet_bg="#1a1730",
)

# Run needs measured loss paths, which the studio no longer sends, so the viewer offers the four levels it can draw
LEVELS = ('Model', 'Window', 'Head', 'Weights')
TEXT, SMALL = .036, .028  # control text, captions and hints
PAD, GAP = .03, .015  # inside a panel; between stacked controls
PANEL_W = .62  # the side panels
CONTROL_W = PANEL_W - 2 * PAD  # every menu and side-panel button
CONTROL_H = 1.6 * TEXT  # frame from .5 below the baseline to 1.1 above, in text units
ROW_H = .105  # one labelled control: caption, then the control
MARKER = .6  # what DirectOptionMenu adds to a frame for its popup marker: .4 wide plus a .1 border each side
KEYS = ('Drag: look around | Wheel: zoom | W A S D, Q E: move | Tab: next block | Up / Down: token | '
        'Page Up / Down: head | [ ] , .: pan the matrix | I J K L: pick a cell | Esc: whole model')


def rgba(name, alpha=1.):
    """A theme colour (a key of C, or '#rrggbb') as the float tuple Panda3D takes."""
    value = C.get(name, name).lstrip('#')
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (float(alpha),)


# one colour per meaning in the 3D scene, so activations, the token mixer and the floor never blend into one cyan
SCENE = {
    'activation': rgba('blue', .85),  # residual-stream size per token, and the Model level's bins
    'mixer': rgba('pink', .85),  # attention beams and arcs, the conv fan, the recurrent ribbon
    'channel': rgba('violet', .7),  # the block's MLP
    'token': rgba('amber'),  # the selected token
    'lane': rgba('border'),  # the floor line under each token
    'gate': rgba('border'),  # the block's panel; its alpha says whether it is selected
    'positive': rgba('amber', .95), 'negative': rgba('blue', .95),  # a matrix value's sign at Head and Weights
    'masked': rgba('panel2'), 'picked': rgba('text'),  # a key the query may not see; the cell I J K L picked
}


def matrix_legend(level):
    """How to read the Head and Weights bars, shown under the HUD's first line."""
    legend = 'amber = positive, blue = negative, taller = larger'
    if level == 'Head':
        return f'rows = queries, columns = keys | {legend} | flat dark = masked (a later key)'
    return f'{legend} | white = the picked cell'


def button_colors(primary=False):
    """frameColor per DirectButton state (ready, pressed, hover, disabled): flat, the app's button tones."""
    if primary:
        return rgba('violet'), rgba('#b3a4ff'), rgba('#b3a4ff'), rgba('panel2')
    return rgba('panel2'), rgba('border'), rgba('border'), rgba('panel2')


def text_colors(primary=False):
    """Text colour per DirectButton state; disabled text is muted, like the app's disabled buttons."""
    ready = rgba('bg') if primary else rgba('text')
    return ready, ready, ready, rgba('muted')


def menu_frame(width=CONTROL_W):
    """frameSize (text units) for a DirectOptionMenu whose finished width, popup marker included, is `width`."""
    return (-.4, width / TEXT - .4 - MARKER, -.5, 1.1)


def button_frame(width=CONTROL_W):
    """frameSize (text units) for a button `width` wide, centred on its position."""
    half = width / TEXT / 2
    return (-half, half, -.5, 1.1)


def view_rows(level, mode, matrices):
    """The view panel's rows, top to bottom: zoom and mode always; the block's mixer in Build mode; the head's step at
    Head; what to show and which matrix at Weights (the matrix only once a snapshot has some)."""
    rows = ['level', 'mode']
    if mode == 'Build':
        rows.append('mixer')
    if level == 'Head':
        rows.append('stage')
    if level == 'Weights':
        rows.append('tile')
        if matrices:
            rows.append('matrix')
    return rows


def captions(layer):
    """Each row's caption."""
    return {'level': 'ZOOM', 'mode': 'MODE', 'mixer': f'BLOCK {layer + 1} TOKEN MIXER', 'stage': 'HEAD STEP',
            'tile': 'SHOW', 'matrix': f'MATRIX (BLOCK {layer + 1})'}


def view_layout(count):
    """The view panel with `count` rows, top edge at 0: [(caption baseline z, control baseline z)] and the bottom edge."""
    rows = [(-PAD - i * ROW_H - .022, -PAD - i * ROW_H - .03 - 1.1 * TEXT) for i in range(count)]
    return rows, -(PAD + (count - 1) * ROW_H + .03 + CONTROL_H + PAD)


def stack_layout(count):
    """A panel of `count` stacked buttons, top edge at 0: [baseline z] and the bottom edge."""
    rows = [-PAD - i * (CONTROL_H + GAP) - 1.1 * TEXT for i in range(count)]
    return rows, -(2 * PAD + count * CONTROL_H + max(0, count - 1) * GAP)


def block_matrices(names, layer):
    """The matrices the menu offers: the selected block's, plus the ones outside any block (token, position embeddings).
    A deep model has dozens; one block's fit on screen."""
    return [n for n in names if not n.startswith('block') or n.startswith(f'block{layer}.')]


def pick_matrix(names, layer, current):
    """The matrix to show after the block or the snapshot changes: the current one if it is still offered, else the
    block's first."""
    offered = block_matrices(names, layer)
    if current in offered:
        return current
    return next((n for n in offered if n.startswith('block')), offered[0] if offered else None)


def playback(count, index, playing):
    """The snapshot controls: {'show': bool, 'play': label, 'position': text, 'previous': bool, 'next': bool}.
    The buttons are hidden until snapshots are loaded; the position line says why."""
    if not count:
        return {'show': False, 'play': 'Play', 'position': 'No snapshots loaded.', 'previous': False, 'next': False}
    return {'show': True, 'play': 'Pause' if playing else 'Play', 'position': f'Snapshot {index + 1} of {count}',
            'previous': index > 0, 'next': index + 1 < count}


def snapshot_buttons(linked, probing):
    """{'show': bool, 'probe': label, 'probe_enabled': bool, 'note': text}. Both buttons ask the studio, so a demo viewer
    opened on its own hides them and says why; the step button is disabled, and says so, while its step runs."""
    if not linked:
        return {'show': False, 'probe': 'Run one real step', 'probe_enabled': False,
                'note': 'Demo model. Open the viewer from the studio to load your snapshots.'}
    return {'show': True, 'probe': 'Running one step...' if probing else 'Run one real step',
            'probe_enabled': not probing, 'note': ''}


def toggles(level, snapshots, auto_orbit):
    """The on/off buttons that change something now: motion only moves the replay pulse (and the demo's slow orbit);
    numbers are drawn at every level but Model."""
    shown = []
    if snapshots or auto_orbit:
        shown.append('motion')
    if level != 'Model':
        shown.append('numbers')
    return shown


def toggle_text(name, on):
    return f"{name}: {'on' if on else 'off'}"


def place_labels(items, height, gap=.01):
    """Which block labels show: items are (x, y, width) in screen units, the label that matters most first (the selected
    block, then the nearest), None for one off screen. Each label stays on its own block so it turns with the scene; one
    that would cover a label already shown is hidden rather than moved. Returns True (show) or False each."""
    placed, out = [], []
    for item in items:
        if item is None:
            out.append(False)
            continue
        x, y, width = item
        clear = all(abs(x - px) >= (width + pw) / 2 + gap - 1e-9 or abs(y - py) >= height + gap - 1e-9  # just touching is apart
                    for px, py, pw in placed)
        if clear:
            placed.append(item)
        out.append(clear)
    return out


def pass_seconds(blocks):
    """How long Play shows one snapshot: the replay pulse runs forward through every block, then back, two blocks a
    second, before Play moves to the next snapshot (or stops after the last, so one real step still plays in full)."""
    return max(1, blocks)


def empty_matrix_note(level, has_snapshot, layer, mixer, has_heads):
    """What the Head or Weights level says when it has no matrix to draw for the selected block."""
    if not has_snapshot:
        return 'Load snapshots, or run one real step, to see these values.'
    if level == 'Head' and not has_heads:
        return f'Block {layer + 1} mixes tokens with {mixer}, which has no attention heads. Tab moves to the next block.'
    return f'This snapshot has no {level.lower()} values for block {layer + 1}.'


def camera_view(level, blocks, rows=16):
    """(target (x, y, z), distance) that frames a level: the column of blocks at Model and Window, the matrix at Head and
    Weights (drawn from y = 0 down in steps of .4)."""
    if level in ('Head', 'Weights'):
        return (0., -(max(1, rows) - 1) * .2, 0.), max(8., rows * .75)
    return (0., -max(1, blocks - 1) * 1.5, 1.), max(22., blocks * 3.5)
