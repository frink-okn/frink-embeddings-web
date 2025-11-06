from frink_embeddings_web.context import AppContext


def update_graph_catalog(ctx: AppContext):
    res = ctx.client.query_points_groups(
        collection_name=ctx.collection,
        with_vectors=False,
        group_by="graph",
        group_size=1,
        limit=100,
        timeout=None,
    )
    groups = [str(group.id).strip() for group in res.groups]
    with ctx.graph_catalog.open("w") as fp:
        fp.write("\n".join(groups))
    return groups
