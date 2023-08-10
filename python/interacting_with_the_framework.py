from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.COMMAND_PROFILE)
from tvb.interfaces.command.lab import *

list_projects()

proj = new_project("sandbox")

list_datatypes(proj.id)

import os
import tvb_data
p = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity/connectivity_66.zip')
import_conn_zip(proj.id, p)

list_datatypes(proj.id)

conn, = load_dt(16)
conn

params = dict(connectivity=conn)
sim_op = fire_simulation(project_id=proj.id, connectivity=conn.gid, simulation_length=100)

list_datatypes(proj.id)

from tvb.core.entities.storage import dao
surface,  = load_dt(39)
surface = dao.get_datatype_by_gid(surface.gid)
surface

surface.hemisphere_mask = [1 if v[0]< 0 else 0 for v in surface.vertices]
surface.bi_hemispheric = True
surface.configure()
surface.persist_full_metadata()
surface

from tvb.core.traits.db_events import fill_before_insert
fill_before_insert(_, _, surface)
dao.store_entity(surface)

