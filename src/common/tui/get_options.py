import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from utils.path import apply_project_directory

apply_project_directory()


def get_options_by_textual(items, title=None):
    from textual import events, on
    from textual.app import App
    from textual.widgets import Label, Footer, Header, ListView, ListItem

    class OptionsApp(App):
        BINDINGS = [
            ("enter", "select", "Select"),
            ("s", "select", "Select"),
            ("q", "sys_exit", "Quit"),
        ]

        def __init__(self, texts, title=None):
            super().__init__()
            self.texts = texts
            self.current_selected = None
            if title is not None:
                self.title = title

        def compose(self):
            yield Header(True)
            yield ListView(
                *[ListItem(Label(item), name=item) for item in self.texts],
                id='option_list'
            )
            yield Footer()

        def action_select(self):
            self.current_selected = self.texts[self.query_one(ListView).index]
            self.exit()

        def action_sys_exit(self):
            self.current_selected = None
            self.exit()

        def on_key(self, event: events.Key):
            if event.key == 'enter':
                self.action_select()


    app = OptionsApp(list(items), title=title)

    app.run()
    return app.current_selected


def get_options_by_input(items, title=None):
    items = list(items)
    print('[#] ', 'select one' if title is None else title)
    print('\n'.join([f'* {i + 1}. {f}' for i, f in enumerate(items)]))
    print('* 0. exit')

    num = int(input('>>> '))

    if num < 0 or num > len(items):
        return None

    return None if num == 0 else items[num - 1]


def get_options(items, title=None, render='textual', is_exit=False, default=None):
    items = list(items)

    if len(items) != 0:

        try:
            si = globals()[f'get_options_by_{render}'](items, title)
        except ImportError:
            si = get_options_by_input(items, title)

        if si is not None:
            return si

    if default is not None:
        return default

    if is_exit == True:
        sys.exit(1)

    return None
