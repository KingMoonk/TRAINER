"""Saved revisions, checkpoint weights and backups."""
from __future__ import annotations

from tkinter import ttk

from studio_net_common import NetworkView


class VersionsView(NetworkView):
    title = "Versions"

    def __init__(self, parent, host):
        super().__init__(parent, host)
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True)
        self.trees = {}
        for name, columns in (("History", ("revision", "change", "saved")),
                              ("Weights", ("version", "revision", "step", "loss")),
                              ("Backups", ("created", "label", "size"))):
            page = ttk.Frame(self.tabs, padding=8)
            self.tabs.add(page, text=name)
            tree = ttk.Treeview(page, columns=columns, show="headings")
            for column in columns:
                tree.heading(column, text=column.title())
                tree.column(column, width=140)
            tree.pack(fill="both", expand=True)
            self.trees[name] = tree
            row = ttk.Frame(page)
            row.pack(fill="x", pady=8)
            if name == "History":
                ttk.Button(row, text="Restore selected revision", command=self.restore).pack(side="left")
            elif name == "Weights":
                ttk.Label(page, text="The selected version is the checkpoint Trace, Inside a head, Value map and the scene inspect (none selected: the newest).",
                          style="Muted.TLabel").pack(anchor="w", before=tree)
                ttk.Button(row, text="Use as initial weights", command=self.use_weights).pack(side="left")
                ttk.Button(row, text="Register checkpoint…", command=host.net_register_checkpoint).pack(side="left", padx=8)
            else:
                ttk.Button(row, text="Create backup", command=host.net_backup).pack(side="left")
                ttk.Button(row, text="Restore backup…", command=self.restore_backup).pack(side="left", padx=8)

    def restore(self):
        selected = self.trees["History"].selection()
        if selected:
            self.host.net_restore_revision(int(selected[0]))

    def restore_backup(self):
        """Restore the backup selected here (not the inspector's list); with none selected, ask for a file."""
        self.host.net_restore_backup(tree=self.trees["Backups"])

    def use_weights(self):
        selected = self.trees["Weights"].selection()
        if selected:
            self.host.net_use_weights(int(selected[0]))

    def refresh(self):
        if self.host.arch_id is None:
            return
        picked = self.trees["Weights"].selection()  # the inspection views' checkpoint picker survives a refresh
        for tree in self.trees.values():
            tree.delete(*tree.get_children())
        for row in self.host.arch_store.revisions(self.host.arch_id):
            self.trees["History"].insert("", "end", iid=str(row["number"]),
                                         values=(f"r{row['number']}", row["note"], row["created_at"][:16]))
        for row in self.host.arch_store.weights(self.host.arch_id):
            self.trees["Weights"].insert("", "end", iid=str(row["version"]),
                                         values=(f"v{row['version']}", f"r{row['revision']}", row["step"], row["loss"]))
        if picked and self.trees["Weights"].exists(picked[0]):
            self.trees["Weights"].selection_set(picked[0])
        for row in self.host.arch_store.backups():
            self.trees["Backups"].insert("", "end", iid=row["path"],
                                         values=(row["created_at"][:16], row["label"], f"{row['bytes'] / 1024:.0f} KB"))
