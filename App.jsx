import { useState } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import "./App.css";
import Navbar from "./components/Navbar";
import Sidebar from "./components/Sidebar";

import Dashboard from "./pages/Dashboard";
import Incidents from "./pages/Incidents";
import IncidentDetails from "./pages/IncidentDetails";
import AddIncident from "./pages/AddIncident";
import EditIncident from "./pages/EditIncident";
import Analytics from "./pages/Analytics";
import NlpLab from "./pages/NlpLab";

function App() {

    const [sidebarOpen, setSidebarOpen] = useState(false);

    return (
        <BrowserRouter>

            <div className="app-shell">

                <Navbar
                    onMenuClick={() => setSidebarOpen(!sidebarOpen)}
                />

                <div className="app-body">

                    <Sidebar
                        collapsed={!sidebarOpen}
                        onClose={() => setSidebarOpen(false)}
                    />

                    <main className="app-main">

                        <div className="page-container">

                            <Routes>

                                <Route
                                    path="/"
                                    element={<Dashboard />}
                                />

                                <Route
                                    path="/incidents"
                                    element={<Incidents />}
                                />

                                <Route
                                    path="/incidents/:id"
                                    element={<IncidentDetails />}
                                />

                                <Route
                                    path="/incidents/:id/edit"
                                    element={<EditIncident />}
                                />

                                <Route
                                    path="/add-incident"
                                    element={<AddIncident />}
                                />

                                <Route
                                    path="/analytics"
                                    element={<Analytics />}
                                />

                                <Route
                                    path="/nlp-lab"
                                    element={<NlpLab />}
                                />

                            </Routes>

                        </div>

                    </main>

                </div>

            </div>

        </BrowserRouter>
    );
}

export default App;