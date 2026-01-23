"use client";

export function ContextPanel() {
    return (
        <aside className="w-80 bg-white border-l border-[#E5E7EB] h-screen fixed right-0 top-0 p-6 hidden lg:block">
            <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wider mb-4">
                Context
            </h3>

            <div className="space-y-4">
                <div className="nexus-card bg-gray-50">
                    <p className="text-sm text-gray-500">System Status</p>
                    <div className="flex items-center gap-2 mt-2">
                        <span className="w-2 h-2 rounded-full bg-green-500"></span>
                        <span className="text-sm font-medium text-gray-900">Operational</span>
                    </div>
                </div>

                <div className="text-xs text-gray-400">
                    Select an item to view details here.
                </div>
            </div>
        </aside>
    );
}
