import datetime
from datetime import timezone
from typing import Any, TypedDict

from sqlalchemy import and_, case, delete, func, select, update
from sqlalchemy.orm import Session as SessionType
from sqlalchemy.orm import sessionmaker

from activetigger.datamodels import AnnotationModel, FeatureDescriptionModelOut
from activetigger.db import DBException
from activetigger.db.models import Annotations, Auths, Features, Models, Projects, Schemes, Tokens


class Codebook(TypedDict):
    codebook: str
    time: str


class ProjectsService:
    Session: sessionmaker[SessionType]

    def __init__(self, sessionmaker: sessionmaker[SessionType]):
        self.Session = sessionmaker

    def get_project(self, project_slug: str):
        session = self.Session()
        project = session.query(Projects).filter_by(project_slug=project_slug).first()
        session.close()
        if project:
            return project.__dict__
        else:
            return None

    def add_project(self, project_slug: str, parameters: dict[str, Any], user_name: str) -> str:
        with self.Session.begin() as session:
            now = datetime.datetime.now(timezone.utc)
            project = Projects(
                project_slug=project_slug,
                parameters=parameters,
                time_created=now,
                time_modified=now,
                user_name=user_name,
            )
            session.add(project)
        return project_slug

    def update_project(self, project_slug: str, parameters: dict[str, Any]):
        with self.Session.begin() as session:
            project = session.query(Projects).filter_by(project_slug=project_slug).first()
            if project is None:
                raise DBException("Project not found")

            project.time_modified = datetime.datetime.now(timezone.utc)
            project.parameters = parameters

    def existing_projects(self) -> list[str]:
        session = self.Session()
        projects = session.query(Projects).all()
        session.close()
        return [project.project_slug for project in projects]

    def add_token(self, token: str, status: str):
        with self.Session.begin() as session:
            new_token = Tokens(
                token=token, status=status, time_created=datetime.datetime.now(timezone.utc)
            )
            session.add(new_token)

    def get_token_status(self, token: str):
        with self.Session() as session:
            found_token = session.scalars(select(Tokens).filter_by(token=token)).first()
            if found_token is None:
                raise DBException("Token not found")
            return found_token.status

    def revoke_token(self, token: str):
        with self.Session.begin() as session:
            _ = session.execute(
                update(Tokens)
                .filter_by(token=token)
                .values(time_revoked=datetime.datetime.now(timezone.utc), status="revoked")
            )

    def add_scheme(
        self,
        project_slug: str,
        name: str,
        labels: list[str],
        kind: str,
        user_name: str,
        codebook: str = "# Guidelines\nWrite down your guidelines here.",
    ):
        if not labels:
            labels = []
        params = {"labels": labels, "codebook": codebook, "kind": kind}
        with self.Session.begin() as session:
            scheme = Schemes(
                project_slug=project_slug,
                name=name,
                params=params,
                user_name=user_name,
                time_created=datetime.datetime.now(timezone.utc),
                time_modified=datetime.datetime.now(timezone.utc),
            )
            session.add(scheme)

    def update_scheme_labels(self, project_slug: str, name: str, labels: list[str]):
        """
        Update the labels in the database
        """
        with self.Session.begin() as session:
            scheme = session.scalars(
                select(Schemes).filter_by(project_slug=project_slug, name=name)
            ).first()
            if scheme is None:
                raise DBException("Scheme not found")
            params = scheme.params.copy()
            params["labels"] = labels
            scheme.params = params
            scheme.time_modified = datetime.datetime.now(timezone.utc)

    def update_scheme_codebook(self, project_slug: str, scheme: str, codebook: str) -> None:
        """
        Update the codebook in the database
        """
        with self.Session.begin() as session:
            result_scheme = session.scalars(
                select(Schemes).filter_by(project_slug=project_slug, name=scheme)
            ).first()
            if result_scheme is None:
                raise DBException("Scheme not found")
            params = result_scheme.params.copy()
            params["codebook"] = codebook
            result_scheme.params = params
            result_scheme.time_modified = datetime.datetime.now(timezone.utc)

    def get_scheme_codebook(self, project_slug: str, name: str) -> Codebook:
        with self.Session() as session:
            scheme = session.scalars(
                select(Schemes).filter_by(project_slug=project_slug, name=name)
            ).first()
            if scheme is None:
                raise DBException("Scheme not found")
            return {
                "codebook": scheme.params["codebook"],
                "time": str(scheme.time_modified),
            }

    def delete_project(self, project_slug: str) -> None:
        with self.Session.begin() as session:
            project = session.scalars(select(Projects).filter_by(project_slug=project_slug)).first()
            if project is None:
                return None
            session.delete(project)

    def get_project_auth(self, project_slug: str):
        with self.Session() as session:
            auth = session.scalars(select(Auths).filter_by(project_slug=project_slug)).all()
            return {el.user_name: el.status for el in auth}

    def add_auth(self, project_slug: str, user_name: str, status: str):
        with self.Session.begin() as session:
            auth = session.scalars(
                select(Auths).filter_by(project_slug=project_slug, user_name=user_name)
            ).first()
            if auth is not None:
                auth.status = status
            else:
                auth = Auths(project_slug=project_slug, user_name=user_name, status=status)
                session.add(auth)

    def delete_auth(self, project_slug: str, user_name: str):
        with self.Session.begin() as session:
            _ = session.execute(
                delete(Auths).filter_by(project_slug=project_slug, user_name=user_name)
            )

    def get_user_auth_projects(self, user_name: str, auth: str | None = None) -> list:
        """
        Projects user can access (auth)
        """
        with self.Session() as session:
            query = (
                select(
                    Auths.project_slug,
                    Auths.status,
                    Projects.parameters,
                    Projects.user_name,
                    Projects.time_created,
                )
                .join(Auths.project)
                .where(Auths.user_name == user_name)
            )
            if auth is not None:
                query = query.where(Auths.status == auth)

            result = session.execute(query).all()
            return list(result)

    def get_user_auth(self, user_name: str, project_slug: str | None = None):
        session = self.Session()
        if project_slug is None:
            result = (
                session.query(Auths.user_name, Auths.status)
                .filter(Auths.user_name == user_name)
                .all()
            )
        else:
            result = (
                session.query(Auths.user_name, Auths.status)
                .filter(Auths.user_name == user_name, Auths.project_slug == project_slug)
                .all()
            )
        session.close()
        return [[row[0], row[1]] for row in result]

    def get_scheme_elements(
        self, project_slug: str, scheme: str, dataset: list[str], user: str | None = None
    ) -> list[list]:
        """
        Get last annotation for each element id for a project/scheme
        """
        with self.Session() as session:
            # Subquery: get max time per element_id with filters
            if user is not None and user != "all":  # only filter by user if specified
                subq = (
                    select(Annotations.element_id, func.max(Annotations.time).label("time"))
                    .where(
                        Annotations.scheme_name == scheme,
                        Annotations.project_slug == project_slug,
                        Annotations.dataset.in_(dataset),
                        Annotations.user_name == user,
                    )
                    .group_by(Annotations.element_id)
                    .subquery()
                )
            else:  # if no user filter, get all annotations
                subq = (
                    select(Annotations.element_id, func.max(Annotations.time).label("time"))
                    .where(
                        Annotations.scheme_name == scheme,
                        Annotations.project_slug == project_slug,
                        Annotations.dataset.in_(dataset),
                    )
                    .group_by(Annotations.element_id)
                    .subquery()
                )

            # Main query: join back on element_id and time to get full rows
            stmt = (
                select(
                    Annotations.scheme_name,
                    Annotations.element_id,
                    Annotations.dataset,
                    Annotations.annotation,
                    Annotations.user_name,
                    Annotations.time,
                    Annotations.comment,
                )
                .join(
                    subq,
                    and_(
                        Annotations.scheme_name == scheme,
                        Annotations.element_id == subq.c.element_id,
                        Annotations.time == subq.c.time,
                    ),
                )
                .where(subq.c.element_id.is_not(None), subq.c.time.is_not(None))
            )
            results = session.execute(stmt)

            # Execute the query and fetch all results
            return [
                [
                    row.element_id,
                    row.dataset,
                    row.annotation,
                    row.user_name,
                    row.time,
                    row.comment,
                ]
                for row in results
            ]

    def get_recent_annotations(
        self, project_slug: str, user_name: str, scheme: str, limit: int, dataset: str = "train"
    ):
        with self.Session() as session:
            stmt = (
                select(Annotations.element_id, Annotations.time)
                .filter_by(
                    project_slug=project_slug,
                    scheme_name=scheme,
                    dataset=dataset,
                )
                .order_by(Annotations.time.desc())
                .limit(limit)
                .distinct()
            )
            if user_name != "all":
                stmt = stmt.filter_by(user_name=user_name)
            recent_annotations = session.execute(stmt)
            return [u[0] for u in recent_annotations]

    def get_annotations_by_element(
        self, project_slug: str, scheme: str, element_id: str, limit: int = 10
    ) -> list[AnnotationModel]:
        with self.Session() as session:
            annotations = (
                session.execute(
                    select(Annotations)
                    .filter_by(
                        project_slug=project_slug,
                        scheme_name=scheme,
                        element_id=element_id,
                    )
                    .order_by(Annotations.time.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
            return [
                AnnotationModel(
                    project_slug=project_slug,
                    scheme=r.scheme_name,
                    element_id=r.element_id,
                    label=r.annotation,
                    dataset=r.dataset,
                    comment=r.comment,
                    selection="by element",
                    time=r.time,
                    user=r.user_name,
                )
                for r in annotations
            ]

    def delete_annotations_evalset(self, project_slug: str, dataset: str):
        """
        Delete all annotations for the eval set in a project.
        This is used when the eval set is deleted or changed.
        """
        with self.Session.begin() as session:
            _ = session.execute(
                delete(Annotations).filter_by(project_slug=project_slug, dataset=dataset)
            )

    def rename_label(self, project_slug: str, scheme: str, former_label: str, new_label: str):
        """
        Rename a label in the database
        - get all annotations with the former label and update them with the new label
        - update the scheme labels
        """
        with self.Session.begin() as session:
            # update annotations
            _ = session.execute(
                update(Annotations)
                .filter_by(project_slug=project_slug, scheme_name=scheme, annotation=former_label)
                .values(
                    annotation=new_label,
                    comment=case(
                        # If comment exists, append the rename info
                        (
                            Annotations.comment.isnot(None),
                            func.concat(
                                Annotations.comment,
                                f" | Renamed from '{former_label}' to '{new_label}'",
                            ),
                        ),
                        # If no comment, create new one
                        else_=f"Renamed from '{former_label}' to '{new_label}'",
                    ),
                )
            )

    def add_annotations(
        self,
        dataset: str,
        user_name: str,
        project_slug: str,
        scheme: str,
        elements: list[dict],  # [{"element_id": str, "annotation": str, "comment": str}]
        selection: str = "not defined",
    ):
        session = self.Session()
        for e in elements:
            annotation = Annotations(
                time=datetime.datetime.now(timezone.utc),
                dataset=dataset,
                user_name=e.get(
                    "user_name", user_name
                ),  # allow overriding user_name for each annotation
                project_slug=project_slug,
                element_id=e["element_id"],
                scheme_name=scheme,
                annotation=e["annotation"],
                comment=e["comment"],
                selection=selection,
            )
            session.add(annotation)
        session.commit()
        session.close()

    def add_annotation(
        self,
        dataset: str,
        user_name: str,
        project_slug: str,
        element_id: str,
        scheme: str,
        annotation: str | None,
        comment: str | None = "",
        selection: str | None = None,
    ):
        with self.Session.begin() as session:
            new_annotation = Annotations(
                time=datetime.datetime.now(timezone.utc),
                dataset=dataset,
                user_name=user_name,
                project_slug=project_slug,
                element_id=element_id,
                scheme_name=scheme,
                annotation=annotation,
                comment=comment,
                selection=selection,
            )
            session.add(new_annotation)

    def available_schemes(self, project_slug: str):
        with self.Session() as session:
            schemes = session.execute(
                select(Schemes.name, Schemes.params).filter_by(project_slug=project_slug)
            ).all()
        r = []
        for s in schemes:
            params = s.params
            kind = params["kind"] if "kind" in params else "multiclass"  # temporary hack
            r.append(
                {
                    "name": s.name,
                    "labels": params["labels"],
                    "codebook": params["codebook"],
                    "kind": kind,
                }
            )
        return r

    def delete_scheme(self, project_slug: str, name: str):
        with self.Session.begin() as session:
            _ = session.execute(delete(Schemes).filter_by(name=name, project_slug=project_slug))

    def duplicate_scheme(self, project_slug: str, name: str, new_name: str, user_name: str) -> None:
        """
        Duplicate a scheme in the database
        - copy the scheme with the new name
        - change the scheme in annotations / models
        """
        with self.Session.begin() as session:
            # get the existing scheme
            old_scheme = session.scalars(
                select(Schemes).filter_by(project_slug=project_slug, name=name)
            ).first()
            if old_scheme is None:
                raise DBException("Scheme not found")

            # Check if new_name already exists (to avoid duplicate key errors)
            existing = session.scalars(
                select(Schemes).filter_by(project_slug=project_slug, name=new_name)
            ).first()
            if existing is not None:
                raise DBException("A scheme with the new name already exists")

            # Create new scheme entry
            new_scheme = Schemes(
                **{
                    c.name: getattr(old_scheme, c.name)
                    for c in Schemes.__table__.columns
                    if c.name not in ("name", "time_modified", "user_name")
                },
                name=new_name,
                time_modified=datetime.datetime.now(timezone.utc),
                user_name=user_name,
            )

            # add it
            session.add(new_scheme)

            # get all the annotations to duplicate
            annotations_to_copy = session.scalars(
                select(Annotations).filter_by(project_slug=project_slug, scheme_name=name)
            ).all()

            # Create all new annotation objects in a list
            new_annotations = [
                Annotations(
                    time=ann.time,
                    dataset=ann.dataset,
                    user_name=ann.user_name,
                    project_slug=ann.project_slug,
                    element_id=ann.element_id,
                    scheme_name=new_name,
                    annotation=ann.annotation,
                    comment=ann.comment,
                    selection=ann.selection,
                )
                for ann in annotations_to_copy
            ]

            # Bulk add them to the session
            session.add_all(new_annotations)

    def rename_scheme(self, project_slug: str, old_name: str, new_name: str) -> None:
        """
        Rename a scheme in the database
        - copy the scheme with the new name
        - change the scheme in annotations / models
        - delete the old scheme
        """
        with self.Session.begin() as session:
            # get the scheme
            old_scheme = session.scalars(
                select(Schemes).filter_by(project_slug=project_slug, name=old_name)
            ).first()
            if old_scheme is None:
                raise DBException("Scheme not found")

            # Check if new_name already exists (to avoid duplicate key errors)
            existing = session.scalars(
                select(Schemes).filter_by(project_slug=project_slug, name=new_name)
            ).first()
            if existing is not None:
                raise DBException("A scheme with the new name already exists")

            # Create new scheme entry
            new_scheme = Schemes(
                **{
                    c.name: getattr(old_scheme, c.name)
                    for c in Schemes.__table__.columns
                    if c.name not in ("name", "time_modified")
                },
                name=new_name,
                time_modified=datetime.datetime.now(timezone.utc),
            )

            # add it
            session.add(new_scheme)

            # Update references in annotations
            session.execute(
                update(Annotations)
                .filter_by(project_slug=project_slug, scheme_name=old_name)
                .values(scheme_name=new_name)
            )

            # Update references in models
            session.execute(
                update(Models)
                .filter_by(project_slug=project_slug, name=old_name)
                .values(name=new_name)
            )

            # Delete the old scheme
            session.delete(old_scheme)

    def get_table_annotations_users(self, project_slug: str, scheme: str, dataset: str):
        with self.Session() as session:
            subquery = (
                select(
                    Annotations.element_id,
                    Annotations.user_name,
                    func.max(Annotations.time).label("last_timestamp"),
                )
                .filter_by(project_slug=project_slug, scheme_name=scheme, dataset=dataset)
                .group_by(Annotations.element_id, Annotations.user_name)
                .subquery()
            )
            query = select(
                Annotations.element_id,
                Annotations.annotation,
                Annotations.user_name,
                Annotations.time,
                Annotations.dataset,
            ).join(
                subquery,
                and_(
                    Annotations.element_id == subquery.c.element_id,
                    Annotations.user_name == subquery.c.user_name,
                    Annotations.time == subquery.c.last_timestamp,
                ),
            )

            results = session.execute(query).fetchall()
            return [
                [row.element_id, row.annotation, row.user_name, row.time, row.dataset]
                for row in results
            ]

    # feature management

    def add_feature(
        self,
        project_slug: str,
        kind: str,
        name: str,
        parameters: dict[str, Any],
        user_name: str,
        data: list | None = None,
    ):
        with self.Session.begin() as session:
            feature = Features(
                project_slug=project_slug,
                time=datetime.datetime.now(timezone.utc),
                kind=kind,
                name=name,
                parameters=parameters,
                user_name=user_name,
                data=data,
            )
            session.add(feature)

    def delete_feature(self, project_slug: str, name: str):
        session = self.Session()
        session.query(Features).filter(
            Features.name == name, Features.project_slug == project_slug
        ).delete()
        session.commit()
        session.close()

    def delete_project_features(self, project_slug: str):
        """
        Delete all features for a project
        """
        with self.Session.begin() as session:
            session.query(Features).filter(Features.project_slug == project_slug).delete()

    def get_feature(self, project_slug: str, name: str):
        session = self.Session()
        feature = (
            session.query(Features)
            .filter(Features.name == name, Features.project_slug == project_slug)
            .first()
        )
        session.close()
        return feature

    def get_project_features(self, project_slug: str) -> dict[str, FeatureDescriptionModelOut]:
        with self.Session() as session:
            features = session.scalars(select(Features).filter_by(project_slug=project_slug)).all()
            return {
                i.name: FeatureDescriptionModelOut(
                    time=i.time.strftime("%Y-%m-%d %H:%M:%S"),
                    kind=i.kind,
                    parameters=i.parameters,
                    user=str(i.user_name),  # check
                    name=i.name,
                )
                for i in features
            }
