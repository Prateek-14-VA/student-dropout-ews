"""Quick DB check — list tables, test insert + query."""
import sqlite3
from app import create_app
from app.models import db, User, Student


def list_tables():
    conn = sqlite3.connect('instance/app.db')
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    conn.close()
    return tables


def main():
    print("=" * 60)
    print("DB VERIFICATION")
    print("=" * 60)

    # 1. List tables
    tables = list_tables()
    print(f"\nTables found: {tables}")

    # 2. Insert test data
    app = create_app()
    with app.app_context():
        # Clean previous test data
        User.query.filter_by(username='test_mentor').delete()
        Student.query.filter_by(anon_id='S001').delete()
        db.session.commit()

        u = User(username='test_mentor', email='m@test.com',
                 role='mentor', full_name='Test Mentor')
        u.set_password('pass123')
        s = Student(anon_id='S001', department='MCA', semester=1,
                    gender='Male', age=22, admission_year=2025)
        db.session.add(u)
        db.session.add(s)
        db.session.commit()

        print(f"\nInserted user:    {User.query.first()}")
        print(f"Inserted student: {Student.query.first()}")
        print(f"Password check:   {User.query.first().check_password('pass123')}")

    print("\n" + "=" * 60)
    print("DB OK")
    print("=" * 60)


if __name__ == "__main__":
    main()