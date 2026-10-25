function StatCard({ title, value, description }) {
    return (
        <div className="card shadow-sm h-100">
            <div className="card-body">

                <h6 className="text-muted">
                    {title}
                </h6>

                <h2 className="fw-bold">
                    {value}
                </h2>

                <p className="text-muted mb-0">
                    {description}
                </p>

            </div>
        </div>
    );
}

export default StatCard;