const pool = require("./config/db");

async function testDatabase() {
    try {
        const connection = await pool.getConnection();

        console.log("✅ MySQL connected successfully!");

        const [rows] = await connection.query(
            "SELECT DATABASE() AS database_name"
        );

        console.log("📌 Database:", rows[0].database_name);

        connection.release();
        await pool.end();
    } catch (error) {
        console.error("❌ MySQL connection failed!");
        console.error(error.message);
    }
}

testDatabase();