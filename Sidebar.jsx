import { NavLink } from "react-router-dom";

function Sidebar({ collapsed, onClose }) {

    const links = [
        {
            path: "/",
            icon: "▣",
            label: "Dashboard"
        },
        {
            path: "/incidents",
            icon: "⚠",
            label: "Incidents"
        },
        {
            path: "/add-incident",
            icon: "＋",
            label: "Add Incident"
        },
        {
            path: "/analytics",
            icon: "▥",
            label: "Analytics"
        },
        {
            path: "/nlp-lab",
            icon: "⚡",
            label: "AI Threat Intelligence"
        }
    ];

    return (
        <aside className={`sidebar ${collapsed ? "sidebar-collapsed" : ""}`}>

            <div className="sidebar-heading">
                <span>MENU</span>

                <button
                    className="sidebar-close"
                    onClick={onClose}
                >
                    ×
                </button>
            </div>

            <nav className="sidebar-nav">

                {links.map((link) => (

                    <NavLink
                        key={link.path}
                        to={link.path}
                        onClick={onClose}
                        className={({ isActive }) =>
                            `sidebar-link ${isActive ? "active" : ""}`
                        }
                    >

                        <span className="sidebar-icon">
                            {link.icon}
                        </span>

                        <span className="sidebar-label">
                            {link.label}
                        </span>

                    </NavLink>

                ))}

            </nav>

            <div className="sidebar-footer">

                <div className="security-indicator">
                    <span className="status-dot"></span>

                    <div>
                        <strong>Security</strong>
                        <small>Monitoring Active</small>
                    </div>
                </div>

                <div className="sidebar-version">
                    CyberSentinel v1.0
                </div>

            </div>

        </aside>
    );
}

export default Sidebar;