// Variant 43 of javascript example 1
const fetchUser = async (id) => {
    const response = await fetch(`/api/users/${id}`);
    return response.json();
};