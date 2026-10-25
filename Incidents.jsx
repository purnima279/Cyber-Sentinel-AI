import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import API from "../services/api";

function Incidents() {
    const [incidents, setIncidents] = useState([]);
    const [filteredIncidents, setFilteredIncidents] = useState([]);

    const [search, setSearch] = useState("");
    const [severity, setSeverity] = useState("");
    const [status, setStatus] = useState("");
    const [category, setCategory] = useState("");

    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");

    useEffect(() => {
        fetchIncidents();
    }, []);

    const fetchIncidents = async () => {
        try {
            setLoading(true);

            const response = await API.get("/incidents");

            const data = response.data.data || [];

            setIncidents(data);
            setFilteredIncidents(data);
            setError("");
        } catch (error) {
            console.error("Error fetching incidents:", error);
            setError("Unable to load incidents.");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        let result = [...incidents];

        // Search filter
        if (search.trim() !== "") {
            const searchText = search.toLowerCase();

            result = result.filter((incident) =>
                incident.title?.toLowerCase().includes(searchText) ||
                incident.description?.toLowerCase().includes(searchText) ||
                incident.predicted_category?.toLowerCase().includes(searchText)
            );
        }

        // Severity filter
        if (severity !== "") {
            result = result.filter(
                (incident) => incident.severity === severity
            );
        }

        // Status filter
        if (status !== "") {
            result = result.filter(
                (incident) => incident.status === status
            );
        }

        // Category filter
        if (category !== "") {
            result = result.filter(
                (incident) =>
                    incident.predicted_category === category
            );
        }

        setFilteredIncidents(result);
    }, [search, severity, status, category, incidents]);

    const deleteIncident = async (id) => {
        const confirmDelete = window.confirm(
            "Are you sure you want to delete this incident?"
        );

        if (!confirmDelete) {
            return;
        }

        try {
            await API.delete(`/incidents/${id}`);

            setIncidents((currentIncidents) =>
                currentIncidents.filter(
                    (incident) =>
                        incident.incident_id !== id
                )
            );

            alert("Incident deleted successfully.");
        } catch (error) {
            console.error("Delete error:", error);
            alert("Failed to delete incident.");
        }
    };

    const clearFilters = () => {
        setSearch("");
        setSeverity("");
        setStatus("");
        setCategory("");
    };

    const getSeverityClass = (value) => {
        if (value === "Critical") {
            return "badge bg-danger";
        }

        if (value === "High") {
            return "badge bg-warning text-dark";
        }

        if (value === "Medium") {
            return "badge bg-info text-dark";
        }

        return "badge bg-secondary";
    };

    const getStatusClass = (value) => {
        if (value === "Open") {
            return "badge bg-primary";
        }

        if (value === "Investigating") {
            return "badge bg-warning text-dark";
        }

        if (value === "Resolved") {
            return "badge bg-success";
        }

        return "badge bg-secondary";
    };

    const categories = [
        ...new Set(
            incidents
                .map((incident) => incident.predicted_category)
                .filter(Boolean)
        )
    ];

    return (
        <div>

            {/* Page Heading */}
            <div className="d-flex justify-content-between align-items-center mb-4">

                <div>
                    <h2>Security Incidents</h2>

                    <p className="text-muted mb-0">
                        Manage and analyze cybersecurity incidents
                    </p>
                </div>

                <Link
                    to="/add-incident"
                    className="btn btn-primary"
                >
                    + Add Incident
                </Link>

            </div>


            {/* Error */}
            {error && (
                <div className="alert alert-danger">
                    {error}
                </div>
            )}


            {/* Filters */}
            <div className="card shadow-sm mb-4">

                <div className="card-body">

                    <div className="row g-3">

                        {/* Search */}
                        <div className="col-md-4">

                            <label className="form-label">
                                Search
                            </label>

                            <input
                                type="text"
                                className="form-control"
                                placeholder="Search incidents..."
                                value={search}
                                onChange={(e) =>
                                    setSearch(e.target.value)
                                }
                            />

                        </div>


                        {/* Category */}
                        <div className="col-md-2">

                            <label className="form-label">
                                Category
                            </label>

                            <select
                                className="form-select"
                                value={category}
                                onChange={(e) =>
                                    setCategory(e.target.value)
                                }
                            >

                                <option value="">
                                    All Categories
                                </option>

                                {categories.map((item) => (
                                    <option
                                        key={item}
                                        value={item}
                                    >
                                        {item}
                                    </option>
                                ))}

                            </select>

                        </div>


                        {/* Severity */}
                        <div className="col-md-2">

                            <label className="form-label">
                                Severity
                            </label>

                            <select
                                className="form-select"
                                value={severity}
                                onChange={(e) =>
                                    setSeverity(e.target.value)
                                }
                            >

                                <option value="">
                                    All
                                </option>

                                <option value="Low">
                                    Low
                                </option>

                                <option value="Medium">
                                    Medium
                                </option>

                                <option value="High">
                                    High
                                </option>

                                <option value="Critical">
                                    Critical
                                </option>

                            </select>

                        </div>


                        {/* Status */}
                        <div className="col-md-2">

                            <label className="form-label">
                                Status
                            </label>

                            <select
                                className="form-select"
                                value={status}
                                onChange={(e) =>
                                    setStatus(e.target.value)
                                }
                            >

                                <option value="">
                                    All
                                </option>

                                <option value="Open">
                                    Open
                                </option>

                                <option value="Investigating">
                                    Investigating
                                </option>

                                <option value="Resolved">
                                    Resolved
                                </option>

                                <option value="Closed">
                                    Closed
                                </option>

                            </select>

                        </div>


                        {/* Clear */}
                        <div className="col-md-2 d-flex align-items-end">

                            <button
                                className="btn btn-outline-secondary w-100"
                                onClick={clearFilters}
                            >
                                Clear Filters
                            </button>

                        </div>

                    </div>

                </div>

            </div>


            {/* Results Count */}
            <div className="mb-3">

                <strong>
                    Showing {filteredIncidents.length}
                </strong>

                <span className="text-muted">
                    {" "}of {incidents.length} incidents
                </span>

            </div>


            {/* Loading */}
            {loading ? (

                <div className="alert alert-info">
                    Loading incidents...
                </div>

            ) : (

                /* Incident Table */
                <div className="card shadow-sm">

                    <div className="table-responsive">

                        <table className="table table-hover align-middle mb-0">

                            <thead className="table-dark">

                                <tr>

                                    <th>ID</th>

                                    <th>Title</th>

                                    <th>Category</th>

                                    <th>Severity</th>

                                    <th>Status</th>

                                    <th>Sentiment</th>

                                    <th>Actions</th>

                                </tr>

                            </thead>


                            <tbody>

                                {filteredIncidents.length === 0 ? (

                                    <tr>

                                        <td
                                            colSpan="7"
                                            className="text-center py-4"
                                        >
                                            No incidents found.
                                        </td>

                                    </tr>

                                ) : (

                                    filteredIncidents.map((incident) => (

                                        <tr
                                            key={incident.incident_id}
                                        >

                                            <td>
                                                #{incident.incident_id}
                                            </td>

                                            <td>

                                                <strong>
                                                    {incident.title}
                                                </strong>

                                            </td>

                                            <td>
                                                {incident.predicted_category || "Other"}
                                            </td>

                                            <td>

                                                <span
                                                    className={getSeverityClass(
                                                        incident.severity
                                                    )}
                                                >
                                                    {incident.severity}
                                                </span>

                                            </td>

                                            <td>

                                                <span
                                                    className={getStatusClass(
                                                        incident.status
                                                    )}
                                                >
                                                    {incident.status}
                                                </span>

                                            </td>

                                            <td>
                                                {incident.sentiment || "N/A"}
                                            </td>

                                            <td>

                                                <Link
                                                    to={`/incidents/${incident.incident_id}`}
                                                    className="btn btn-sm btn-outline-primary me-2"
                                                >
                                                    View
                                                </Link>

                                                <button
                                                    className="btn btn-sm btn-outline-danger"
                                                    onClick={() =>
                                                        deleteIncident(
                                                            incident.incident_id
                                                        )
                                                    }
                                                >
                                                    Delete
                                                </button>

                                            </td>

                                        </tr>

                                    ))

                                )}

                            </tbody>

                        </table>

                    </div>

                </div>

            )}

        </div>
    );
}

export default Incidents;