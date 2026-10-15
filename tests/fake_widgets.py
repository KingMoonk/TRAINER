"""Stand-ins for the few Tk widget calls KeyedRows, sync_tree and the row fillers make, so they are tested without a window."""


class Parent:
    """A container that packs like Tk: a new slave goes last, -before/-after move one, a packed slave keeps its place."""

    def __init__(self):
        self.slaves = []

    def pack_slaves(self):
        return list(self.slaves)


class Widget:
    made = 0

    def __init__(self, parent=None, name=None):
        Widget.made += 1
        self.parent, self.name, self.options, self.packed, self.destroyed = parent, name or f".w{Widget.made}", {}, False, False

    def __str__(self):
        return self.name

    def configure(self, **options):
        self.options.update(options)

    def cget(self, key):
        return self.options.get(key, "")

    def pack(self, before=None, after=None, **options):
        slaves = self.parent.slaves
        if before is None and after is None:
            if self not in slaves:
                slaves.append(self)
        else:
            if self in slaves:
                slaves.remove(self)
            slaves.insert(slaves.index(before) if before is not None else slaves.index(after) + 1, self)
        self.packed = True

    def pack_forget(self):
        if self.parent is not None and self in self.parent.slaves:
            self.parent.slaves.remove(self)
        self.packed = False

    def destroy(self):
        self.pack_forget()
        self.destroyed = True


class Tree:
    """The Treeview calls sync_tree makes: children in order, item options per iid, and a log of the calls."""

    def __init__(self, rows=()):
        self.children, self.items, self.calls = [], {}, []
        for iid, options in rows:
            self.insert("", "end", iid=iid, **options)
        self.calls.clear()

    def get_children(self):
        return tuple(self.children)

    def insert(self, parent, index, iid, **options):
        self.calls.append(("insert", iid))
        self.children.insert(len(self.children) if index == "end" else index, iid)
        self.items[iid] = dict(options)

    def item(self, iid, **options):
        self.calls.append(("item", iid))
        self.items[iid].update(options)

    def move(self, iid, parent, index):
        self.calls.append(("move", iid))
        self.children.remove(iid)
        self.children.insert(index, iid)

    def delete(self, *iids):
        self.calls.append(("delete",) + iids)
        for iid in iids:
            self.children.remove(iid)
            del self.items[iid]
